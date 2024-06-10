from .omost_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


import os
import filecmp
import shutil
import __main__


def setup_js():
    extensions_folder = os.path.join(
        os.path.dirname(os.path.realpath(__main__.__file__)),
        "web",
        "extensions",
        "ComfyUI_omost",
    )
    javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

    if not os.path.exists(extensions_folder):
        print("Creating frontend extension folder: " + extensions_folder)
        os.mkdir(extensions_folder)

    result = filecmp.dircmp(javascript_folder, extensions_folder)

    if result.left_only or result.diff_files:
        print("Update to javascripts files detected")
        file_list = list(result.left_only)
        file_list.extend(x for x in result.diff_files if x not in file_list)

        for file in file_list:
            print(f"Copying {file} to extensions folder")
            src_file = os.path.join(javascript_folder, file)
            dst_file = os.path.join(extensions_folder, file)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_file)


setup_js()
