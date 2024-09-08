from layouts.managers.dataset_layout_managers import DatasetTabLayout


def get_layout(config_manager):
    tab_layout = DatasetTabLayout(config_manager)
    return tab_layout.render()
