from plotly.subplots import make_subplots
import plotly.graph_objects as go


def visualize_traker(
    tracker,
    types_error=["train", "val"],
    xaxis_title="epochs",
):
    """Visualize evolution of loss and accuracy during training
    """

    # init fig
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Loss", "Accuracy"],
    )

    # plot loss
    fig.add_traces(
        data=[
            go.Scatter(
                x=[idx + 1 for idx in tracker.keys()],
                y=[epo_meta[f"loss_{type_}"] for epo_meta in tracker.values()],
                name=f"loss_{type_}"
            )
            for type_ in types_error],
        rows=1,
        cols=1
    )

    # plot accuracy
    fig.add_traces(
        data=[
            go.Scatter(
                x=[idx + 1 for idx in tracker.keys()],
                y=[epo_meta[f"acc_{type_}"] for epo_meta in tracker.values()],
                name=f"acc_{type_}"
            )
            for type_ in types_error],
        rows=1,
        cols=2
    )

    # Update x axis properties
    fig.update_xaxes(title_text=xaxis_title, row=1, col=1)
    fig.update_xaxes(title_text=xaxis_title, row=1, col=2)

    return fig
