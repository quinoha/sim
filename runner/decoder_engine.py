import time
import numpy as np
from qec_dataset.reference_decode import decode_shots_pymatching, decode_shots_bposd

def decode_syndromes_parallel (run_spec, circuit, dets_matrix, num_threads=0):
    """
    dets_matrix: The 200,000 shots syndrome matrix that Sinter receives.
    num_thread=0: Parallel decoding that we get through CPU cores.
    """

    t0 = time.perf_counter_ns()

    # 1. win PyMatching
    if run_spec.decoder_id == "pymatching":
        dem = circuit.error_model()
        predictions = decode_shots_pymatching(dem, dets_matrix, num_threads=num_threads)

    elif run_spec.decoder_id == "bposd_fast":
        predictions = decode_shots_bposd(run_spec.code_params, dets_matrix, num_threads=num_threads)

    elif run_spec.decoder_id == "my_custom_gnn":
        pass
        # predictions = my_gnn_model.predict

    elapsed_ns = time.perf_counter_ns() - t0
    return predictions, elapsed_ns
