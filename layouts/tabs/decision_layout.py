from layouts.managers.decision_layout_managers import DecisionTabLayout


def get_layout(config_manager):
    tab_layout = DecisionTabLayout(config_manager)
    return tab_layout.render()
