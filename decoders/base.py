import numpy as np


"""
Base Decoder file with reference code.
"""

def my_gnn_decoder(dets: np.ndarray, run_spec: RunSpec) -> np.ndarray:
    """
    Args:
        dets: shape=(shots, num_detectors), dtype=np.bool_ (0/1 신드롬 문제지)
        run_spec: 해당 런의 물리 파라미터 (code_id, noise_p, rounds 등)
    Returns:
        predictions: shape=(shots, num_observables), dtype=np.bool_ (0/1 예측 정답)
    """
    # GNN (PyTorch/TensorFlow) 또는 RTL Verilator 시뮬레이터 호출
    predictions = model.predict(dets)
    return predictions

# 2. 러너에 등록 (플러그 앤 플레이)
runner = EvaluationRunner(plan=plan)
runner.register_decoder("astra_gnn", my_gnn_decoder)
evidence_bundle = runner.run_all()