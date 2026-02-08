    # Insert logging for temperature annealing progress in make_train_step
        if beta_anneal > 0:
            print(f"Step {step + 1}: Effective beta = {beta_eff:.4f}, T_eff = {T_eff:.4f}")