import pytest
import numpy as np
import decay

def test_simulate_negative_rate():
    with pytest.raises(ValueError):
        decay.simulate(1000, -0.4)

def test_matches_law():
    N0 = 100000
    lam = 0.4
    dt = 0.05
    
    runs = [decay.simulate(N0, lam, dt=dt) for _ in range(20)]
    avg_trajectory = np.mean(runs, axis=0)
    
    t = np.arange(len(avg_trajectory)) * dt
    expected = N0 * np.exp(-lam * t)

    assert avg_trajectory == pytest.approx(expected, rel=5e-2)