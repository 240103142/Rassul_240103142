# MPI + CUDA stencil (Python)

Colab T4-та төрт MPI rank арқылы іске қосуға арналған бастапқы solver.
Бұл нұсқа әлі T4-та орындалмаған; тапсыру алдында Colab нәтижелерімен
тексерілген notebook пен manifest-ті қосу керек.

- `distributed_cuda_solver.py` — локал subdomain, CUDA stencil, halo exchange,
  serialized/overlapped салыстыру және manifest генерациясы.

## Colab-та іске қосу

Алдымен осы репозиторийді Colab-та clone жасап, қажетті пакеттерді орнатыңыз:

```python
!git clone https://github.com/240103142/Rassul_240103142.git
%cd Rassul_240103142
!pip install -q numpy mpi4py numba-cuda
!nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total --format=csv
!mpirun --version | head -n 2
```

Содан кейін нақты GitHub handle-ді беріп іске қосыңыз:

```python
!mpirun --allow-run-as-root --oversubscribe -np 4 \
  python3 distributed_cuda_solver.py 240103142
```

`submission_manifest.json` сәтті орындалғаннан кейін rank 0 арқылы жасалады.
Оған PTX санағы, T4-та өлшенген уақыттар, energy trace және serialized/
overlapped parity нәтижелері кіреді. Бұл мәндерді Colab-тағы нақты іске
қосудан алу керек.

GitHub username және GPU UUID seed-ке пайдаланылады. Толық аты-жөні seed-ке
қажет емес. Solver-дегі source өрісі Q6-дағы dissipative energy trace талабын
көрсету үшін әлсіз локалданған sink ретінде берілген.
