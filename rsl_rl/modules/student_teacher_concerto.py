from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import MLP, EmpiricalNormalization


class StudentTeacherConcerto(nn.Module):
    """Student-Teacher policy where the student uses a Concerto point-cloud encoder + MLP fusion.

    The teacher remains a plain MLP (matching the checkpointed actor). The student fuses the
    flattened policy observations with a point cloud embedding from ConcertoMLP.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = (256, 256),
        teacher_hidden_dims: tuple[int] | list[int] = (256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        # Concerto-specific parameters
        sensor_group: str = "sensor",
        concerto_model_name: str = "concerto_large_outdoor",
        concerto_repo_id: str = "Pointcept/Concerto",
        concerto_ckpt_path: str | None = None,
        concerto_pretrained: bool = True,
        train_backbone: bool = False,
        pc_embed_dim: int = 512,
        pc_grid_size: float = 0.05,
        pool: str = "mean",
        normalize_pc: bool = False,
        use_precomputed_embeddings: bool = False,
        dropout: float = 0.1,
        log_pc_stats_every: int = 0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StudentTeacherConcerto.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False  # Indicates if teacher has been loaded
        self.obs_groups = obs_groups
        self.sensor_group = sensor_group

        # Get observation dimensions
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "Student obs must be flat (B, D)."
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "Teacher obs must be flat (B, D)."
            num_teacher_obs += obs[obs_group].shape[-1]

        # Student: Concerto backbone + MLP head
        from rsl_transformers.models.concerto_mlp import ConcertoMLP  # heavy import guarded to avoid global dependency

        self.student = ConcertoMLP(
            obs_dim=num_student_obs,
            act_dim=num_actions,
            hidden_layers=list(student_hidden_dims),
            activation=activation,
            dropout=dropout,
            concerto_model_name=concerto_model_name,
            concerto_repo_id=concerto_repo_id,
            concerto_ckpt_path=concerto_ckpt_path,
            concerto_pretrained=concerto_pretrained,
            train_backbone=train_backbone,
            pc_embed_dim=pc_embed_dim,
            pool=pool,
            pc_grid_size=pc_grid_size,
            normalize_pc=normalize_pc,
            use_precomputed_embeddings=use_precomputed_embeddings,
            log_pc_stats_every=log_pc_stats_every,
        )
        print(f"Student ConcertoMLP: {self.student}")

        # Student observation normalization (only on flat proprio obs)
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # Teacher: plain MLP
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        print(f"Teacher MLP: {self.teacher}")

        # Teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Use 'scalar' or 'log'.")

        # Action distribution (populated in _update_distribution)
        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None, hidden_states=None) -> None:
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _extract_pc(self, obs: TensorDict) -> torch.Tensor:
        if self.sensor_group not in obs:
            raise ValueError(f"Point cloud observation '{self.sensor_group}' not found in obs.")
        pc = obs[self.sensor_group]
        if pc.ndim == 3 and pc.shape[-1] == 3:
            return pc
        if pc.ndim == 2 and pc.shape[-1] % 3 == 0:
            return pc.view(pc.shape[0], -1, 3)
        raise ValueError(
            f"Point cloud obs must be [B,N,3] or flat [B, N*3]; got shape {tuple(pc.shape)} for '{self.sensor_group}'."
        )

    def _update_distribution(self, obs: TensorDict) -> None:
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        pc = self._extract_pc(obs)
        student_out = self.student(x=student_obs, pc=pc)["output"].squeeze(1)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(student_out)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(student_out)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Use 'scalar' or 'log'.")
        self.distribution = Normal(student_out, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        pc = self._extract_pc(obs)
        return self.student(x=student_obs, pc=pc)["output"].squeeze(1)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        teacher_obs = self.get_teacher_obs(obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        with torch.no_grad():
            return self.teacher(teacher_obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self):
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        # Mirror StudentTeacher loading semantics
        if any("actor" in key for key in state_dict):  # PPO teacher-only checkpoint
            teacher_state_dict = {}
            teacher_obs_norm_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_norm_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_norm_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        elif any("student" in key for key in state_dict):  # Distillation checkpoint
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
