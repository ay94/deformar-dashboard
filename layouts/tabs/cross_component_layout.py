from layouts.managers.cross_component_layout_managers import CrossComponentTab


def get_layout(config_manager):
    tab_layout = CrossComponentTab(config_manager)
    return tab_layout.render()
