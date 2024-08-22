import time
from dash import callback_context, html, no_update
from dash.dependencies import Input, Output, State

from utils.layout_managers import generate_status_table
from utils.plotUtils import get_input_trigger


def register_callbacks(app, data_manager):
    @app.callback(
        [
            Output("data_status_table", "children"),
            Output("load_message", "children"),
            Output("cache_status", "children"),
        ],
        [
            Input("load_variant", "n_clicks"),
            Input("load_data", "n_clicks"),
            Input("clear_cache", "n_clicks"),
        ],
        [State("variant_name", "value")],
    )
    def update_data_and_cache(
        load_variant_clicks, load_data_clicks, clear_cache_clicks, variant_name
    ):
        ctx = callback_context
        triggered_id = get_input_trigger(ctx)

        if triggered_id == "clear_cache" and clear_cache_clicks:
            try:
                data_manager.cache.clear()
                data_manager.variants_data = {}
                return html.Div(), html.Div(), "Cache cleared successfully!"
            except Exception as e:
                return no_update, no_update, f"Error clearing cache: {str(e)}"

        elif triggered_id in ["load_variant", "load_data"]:
            start_time = time.time()
            message = ""
            try:
                if triggered_id == "load_variant" and variant_name:
                    data_manager.load_variant(variant_name)
                    message = f"Data for variant {variant_name} loaded successfully in {time.time() - start_time:.2f} seconds!"
                elif triggered_id == "load_variant" and not variant_name:
                    message = "Please choose a variant or press load data."
                elif triggered_id == "load_data":
                    data_manager.load_data()
                    message = f"Data for all variants loaded successfully in {time.time() - start_time:.2f} seconds!"

                return generate_status_table(data_manager.variants_data), html.Span(message), ""

            except Exception as e:
                return no_update, f"Failed to load data: {str(e)}", ""

        return (
            generate_status_table(data_manager.variants_data),
            no_update,
            no_update,
        )

    return data_manager.variants_data
