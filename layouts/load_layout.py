from utils.load_layout_managers import LoadTabLayout

def get_layout(config_manager):
    tab_layout = LoadTabLayout(config_manager)   
    return tab_layout.render()
    