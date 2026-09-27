import torch
import plotly.graph_objects as go


def plot_curve(
    x: torch.Tensor,
    y: torch.Tensor,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "Simulated data",
    label: str | None = None,
) -> None:
    """
    Plot an interactive curve using Plotly.

    Parameters
    ----------
    x : torch.Tensor
        Values for the x-axis.
    y : torch.Tensor
        Values for the y-axis.
    xlabel : str, default="x"
        Label of the x-axis.
    ylabel : str, default="y"
        Label of the y-axis.
    title : str, default="Simulated data"
        Plot title.
    label : str or None, default=None
        Name of the curve shown in the legend.

    Returns
    -------
    None
        Displays the interactive figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x.detach().cpu().numpy(),
            y=y.detach().cpu().numpy(),
            mode="lines",
            name=label,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
    )

    fig.show()