from shared.ui.dashboard import LiveDashboard


def test_dashboard_compacts_loss_history_across_full_run():
    dashboard = LiveDashboard(total_steps=100)

    dashboard.update(step=0, loss=1.0)
    dashboard.update(step=50, loss=0.5)
    dashboard.update(step=100, loss=0.1)

    assert len(dashboard.metrics.loss_history) == 40
    assert dashboard.metrics.loss_history[0] == 1.0
    assert dashboard.metrics.loss_history[20] == 0.5
    assert dashboard.metrics.loss_history[-1] == 0.1


def test_dashboard_averages_multiple_updates_within_same_bucket():
    dashboard = LiveDashboard(total_steps=100)

    dashboard.update(step=0, loss=1.0)
    dashboard.update(step=1, loss=0.6)

    assert dashboard.metrics.loss_history[0] == 0.8


def test_dashboard_tracks_best_loss_separately_from_bucketed_series():
    dashboard = LiveDashboard(total_steps=100)

    dashboard.update(step=0, loss=1.0)
    dashboard.update(step=1, loss=0.6)
    dashboard.update(step=2, loss=0.9)

    assert dashboard.metrics.best_loss == 0.6
