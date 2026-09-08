from qec_dataset.circuit_builder import build_circuit_from_run_spec
from qec_dataset.sampling import sample_shots_to_files, read_b8_file
from runner.decoder_engine import decode_syndromes_parallel

def run_evaluation_plan(plan, output_dir = "outputs/evidence", quick=False):
    results = []

    for spec in plan.run_specs:
        shots = 1000 if quick else spec.shots
        print(f"[*] Running: {spec.code_id} x {spec.decoder_id} (p={spec.noise_p}, {shots:,} shots)...")

        # build stim circuit from runspec
        circuit = build_circuit_from_run_spec(spec)

        # use sinter multi-process to extract shots and save as .b8 file
        dets_path, obs_path = sample_shots_to_files(circuit, shots=shots, out_dir=output_dir)

        dets_matrix = read_b8_file(dets_path)
        actual_obs = read_b8_file(obs_path)
        predictions, latency_ns = decode_syndromes_parallel(spec, circuit=circuit, dets_matrix=dets_matrix)

        errors = (predictions != actual_obs).any(axis=1).sum()
        ler = errors / shots

        print(f"    -> Completed. num errors: {errors}/{shots} (LER: {ler:.4e}), Latency: {latency_ns / 1e6:.2f} ms")
        results.append({"spec_id": spec.run_spec_id, "ler": ler, "latency_ns": latency_ns})
        
    return results
