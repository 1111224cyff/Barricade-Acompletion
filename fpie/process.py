import os
import random
import cv2
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
from fpie.io import resize_mask_to_fit_target

import numpy as np

from fpie import np_solver

CPU_COUNT = os.cpu_count() or 1
DEFAULT_BACKEND = "numpy"
ALL_BACKEND = ["numpy"]

try:
  from fpie import numba_solver
  ALL_BACKEND += ["numba"]
  DEFAULT_BACKEND = "numba"
except ImportError:
  numba_solver = None  # type: ignore

try:
  from fpie import taichi_solver
  ALL_BACKEND += ["taichi-cpu", "taichi-gpu"]
  DEFAULT_BACKEND = "taichi-gpu"
except ImportError:
  taichi_solver = None  # type: ignore

try:
  from fpie import core_gcc  # type: ignore
  DEFAULT_BACKEND = "gcc"
  ALL_BACKEND.append("gcc")
except ImportError:
  core_gcc = None

try:
  from fpie import core_openmp  # type: ignore
  DEFAULT_BACKEND = "openmp"
  ALL_BACKEND.append("openmp")
except ImportError:
  core_openmp = None

try:
  from mpi4py import MPI

  from fpie import core_mpi  # type: ignore
  ALL_BACKEND.append("mpi")
except ImportError:
  MPI = None  # type: ignore
  core_mpi = None

try:
  from fpie import core_cuda  # type: ignore
  DEFAULT_BACKEND = "cuda"
  ALL_BACKEND.append("cuda")
except ImportError:
  core_cuda = None


class BaseProcessor(ABC):
  """API definition for processor class."""

  def __init__(
    self, gradient: str, rank: int, backend: str, core: Optional[Any]
  ):
    if core is None:
      error_msg = {
        "numpy":
          "Please run `pip install numpy`.",
        "numba":
          "Please run `pip install numba`.",
        "gcc":
          "Please install cmake and gcc in your operating system.",
        "openmp":
          "Please make sure your gcc is compatible with `-fopenmp` option.",
        "mpi":
          "Please install MPI and run `pip install mpi4py`.",
        "cuda":
          "Please make sure nvcc and cuda-related libraries are available.",
        "taichi":
          "Please run `pip install taichi`.",
      }
      print(error_msg[backend.split("-")[0]])

      raise AssertionError(f"Invalid backend {backend}.")

    self.gradient = gradient
    self.rank = rank
    self.backend = backend
    self.core = core
    self.root = rank == 0

  def mixgrad(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if self.gradient == "src":
      return a
    if self.gradient == "avg":
      return (a + b) / 2
    else:
      # Enhanced mixing strategy to prevent transparency issues
      mask = np.abs(a) > np.abs(b)
      result = np.where(mask, a, b)
      return result

  @abstractmethod
  def reset(
    self,
    src: np.ndarray,
    mask: np.ndarray,
    tgt: np.ndarray,
    mask_on_src: Tuple[int, int],
  ):
    pass

  def sync(self) -> None:
    self.core.sync()

  @abstractmethod
  def step(self, iteration: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pass


class EquProcessor(BaseProcessor):
  """PIE Jacobi equation processor."""

  def __init__(
    self,
    gradient: str = "max",
    backend: str = DEFAULT_BACKEND,
    n_cpu: int = CPU_COUNT,
    min_interval: int = 100,
    block_size: int = 1024,
  ):
    core: Optional[Any] = None
    rank = 0

    if backend == "numpy":
      core = np_solver.EquSolver()
    elif backend == "numba" and numba_solver is not None:
      core = numba_solver.EquSolver()
    elif backend == "gcc":
      core = core_gcc.EquSolver()
    elif backend == "openmp" and core_openmp is not None:
      core = core_openmp.EquSolver(n_cpu)
    elif backend == "mpi" and core_mpi is not None:
      core = core_mpi.EquSolver(min_interval)
      rank = MPI.COMM_WORLD.Get_rank()
    elif backend == "cuda" and core_cuda is not None:
      core = core_cuda.EquSolver(block_size)
    elif backend.startswith("taichi") and taichi_solver is not None:
      core = taichi_solver.EquSolver(backend, n_cpu, block_size)

    super().__init__(gradient, rank, backend, core)

  def mask2index(
    self, mask: np.ndarray
  ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    # 返回非0元素的索引
    x, y = np.nonzero(mask)
    # 将所有非零元素的x坐标的个数+1作为max_id
    max_id = x.shape[0] + 1
    # 创建一个形状为max_id行, 3列的零数组 index，用于存储分区的索引信息
    index = np.zeros((max_id, 3))
    # 为mask中每个>0的元素赋予唯一的标识符(其累计和值)，0保持0。具体见xxx_solver.py
    # 输出的ids与输入的mask格式相同
    ids = self.core.partition(mask)
    # 确保所有背景位置在ids中也是0
    ids[mask == 0] = 0  # reserve id=0 for constant
    # ids[x, y]表示获取ids中（x,y）坐标对应的唯一标识符
    # 对标识符排序，形成index列表，则可通过x[index]来访问第index个纵坐标，y类似
    index = ids[x, y].argsort()
    return ids, max_id, x[index], y[index]

  def preprocess_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    取出非零最小包围框，返回该范围的mask
    """
    if len(mask.shape) == 3:
      mask = mask.mean(-1)
    mask = (mask >= 128).astype(np.int32)

    # zero-out edge
    mask[0] = 0
    mask[-1] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0

    # 找出mask图像的非零元素，返回的是行列的索引
    x, y = np.nonzero(mask)
    # 计算mask中非零元素的最值，即确定mask物体的边界（矩形区域）
    x0, x1 = x.min() - 1, x.max() + 2
    y0, y1 = y.min() - 1, y.max() + 2
    # 把包含目标mask的矩形框框出来，其他部分裁去
    cropped_mask = mask[x0:x1, y0:y1]
    return cropped_mask, x0, y0

  def reset(
    self,
    src: np.ndarray,
    mask: np.ndarray,
    tgt: np.ndarray,
    mask_on_src: Tuple[int, int],
  ):
    assert self.root
    # check validity
    # assert 0 <= mask_on_src[0] and 0 <= mask_on_src[1]
    # assert mask_on_src[0] + mask.shape[0] <= src.shape[0]
    # assert mask_on_src[1] + mask.shape[1] <= src.shape[1]
    # assert mask_on_tgt[0] + mask.shape[0] <= tgt.shape[0]
    # assert mask_on_tgt[1] + mask.shape[1] <= tgt.shape[1]

    # 预处理得到包含非零的最小区域框
    cropped_mask, _, _ = self.preprocess_mask(mask)
    # 将mask及src图像缩放至对应大小
    mask, src = resize_mask_to_fit_target(cropped_mask, tgt, src, mask)
    # 重新预处理得到包含非零的最小区域框
    cropped_mask, x0, y0 = self.preprocess_mask(mask)

    # 获取裁剪后mask的宽高
    mask_h, mask_w = cropped_mask.shape[:2]

    # 这里mask2index输入的mask是裁剪后的mask!!!!!
    ids, max_id, index_x, index_y = self.mask2index(cropped_mask)

    # 获取目标图像的高宽
    tgt_h, tgt_w = tgt.shape[:2]

    # 确定矩形左下角的随机位置
    y_left_bottom = random.randint(0, max(0, tgt_w - mask_w - 1))
    x_left_bottom = random.randint(0, max(0, tgt_h - mask_h - 1))

    # 更新mask在源图像和目标图像上的位置，根据遮罩覆盖的实际区域进行调整
    # 因为mask_on_src即(h0,w0)默认为(0,0)，所以mask_on_src就是(x0,y0)
    mask_on_src = (x0 + mask_on_src[0], y0 + mask_on_src[1])
    # 该坐标代表裁剪后蒙版左下角点在目标图中插入的位置
    mask_on_tgt = (x_left_bottom, y_left_bottom)

    # 计算mask在原图和目标图上的坐标
    # 以mask_on_src, mask_on_tgt的坐标作为角点，来插入裁剪后的蒙版的
    src_x, src_y = index_x + mask_on_src[0], index_y + mask_on_src[1]

    # 这个就是目标图上掩码的位置
    tgt_x, tgt_y = index_x + mask_on_tgt[0], index_y + mask_on_tgt[1]

    # 获取原图和目标图的当前像素(C)，以及其上(U)、下(D)、左(L)、右(R)四个方向的邻居像素
    src_C = src[src_x, src_y].astype(np.float32)
    src_U = src[src_x - 1, src_y].astype(np.float32)
    src_D = src[src_x + 1, src_y].astype(np.float32)
    src_L = src[src_x, src_y - 1].astype(np.float32)
    src_R = src[src_x, src_y + 1].astype(np.float32)
    tgt_C = tgt[tgt_x, tgt_y].astype(np.float32)
    tgt_U = tgt[tgt_x - 1, tgt_y].astype(np.float32)
    tgt_D = tgt[tgt_x + 1, tgt_y].astype(np.float32)
    tgt_L = tgt[tgt_x, tgt_y - 1].astype(np.float32)
    tgt_R = tgt[tgt_x, tgt_y + 1].astype(np.float32)

    # 计算梯度
    grad = self.mixgrad(src_C - src_L, tgt_C - tgt_L) \
      + self.mixgrad(src_C - src_R, tgt_C - tgt_R) \
      + self.mixgrad(src_C - src_U, tgt_C - tgt_U) \
      + self.mixgrad(src_C - src_D, tgt_C - tgt_D)

    A = np.zeros((max_id, 4), np.int32)
    X = np.zeros((max_id, 3), np.float32)
    B = np.zeros((max_id, 3), np.float32)

    X[1:] = tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1]]
    # four-way
    A[1:, 0] = ids[index_x - 1, index_y]
    A[1:, 1] = ids[index_x + 1, index_y]
    A[1:, 2] = ids[index_x, index_y - 1]
    A[1:, 3] = ids[index_x, index_y + 1]
    B[1:] = grad
    m = (cropped_mask[index_x - 1, index_y] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0] - 1, index_y + mask_on_tgt[1]]
    m = (cropped_mask[index_x, index_y - 1] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1] - 1]
    m = (cropped_mask[index_x, index_y + 1] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0], index_y + mask_on_tgt[1] + 1]
    m = (cropped_mask[index_x + 1, index_y] == 0).astype(float).reshape(-1, 1)
    B[1:] += m * tgt[index_x + mask_on_tgt[0] + 1, index_y + mask_on_tgt[1]]

    self.tgt = tgt.copy()
    self.tgt_index = (index_x + mask_on_tgt[0], index_y + mask_on_tgt[1])
    self.core.reset(max_id, A, X, B)
    return max_id, tgt_x, tgt_y

  def step(self, iteration: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    result = self.core.step(iteration)
    if self.root:
      x, err = result
      self.tgt[self.tgt_index] = x[1:]
      return self.tgt, err
    return None