from pathlib import Path

from logger import Logger


def build_folder_tree(base_dir):

    logger = Logger("cli").get_logger()
    logger.info(f"Building folder tree for folder: {base_dir}")

    base = Path(base_dir)
    tree = {}
    if not base.exists():
        logger.warning(f"Folder at path {base_dir} doesn't exist.")
        return tree
    for block_dir in base.iterdir():
        if block_dir.is_dir():
            subfolders = {}
            for subdir in block_dir.iterdir():
                if subdir.is_dir():
                    pngs = sorted(subdir.glob("*.png"))
                    subfolders[subdir.name] = pngs
            tree[block_dir.name] = subfolders
    return tree
