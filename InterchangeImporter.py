import unreal
import asyncio
import os
import logging
import multiprocessing
from enum import Enum

'''
Welcome to the IDTA file importer, it uses the UE5 interchange format.
it is not meant to be a final product, it only handles a few formats that I was experimenting with
its easy to extend to handle more. 
Its designed to not be over engineered and be a good template since Ive done this about 30 times. 
Im tired of redoing it. 

To use, setup your python environment in ue5. 
run this as a startup script. 
save files into the appropriate folder.
'''

# Set up logging
logging.basicConfig(filename='importer.log', level=logging.INFO)


class FileType(Enum):
    GLTF = 1
    SUBSTANCE = 2
    HOUDINI = 3


class FileTypeHandler:
    def __init__(self):
        self.handlers = {
            FileType.GLTF: {
                'extensions': ('.gltf', '.glb'),
            },
            FileType.SUBSTANCE: {
                'extensions': ('.sbs', '.sbsar'),
            },
            FileType.HOUDINI: {
                'extensions': ('.hda',),
            }
        }

    def get_file_type(self, file_extension):
        for file_type, info in self.handlers.items():
            if file_extension in info['extensions']:
                return file_type
        return None


class ImportWorkerPool:
    def __init__(self, num_workers=multiprocessing.cpu_count()):
        self.pool = multiprocessing.Pool(num_workers)

    def import_asset(self, file_path, file_type):
        return self.pool.apply_async(AssetImporter.import_file, (file_path, file_type))


class ErrorHandler:
    @staticmethod
    async def retry_import(file_path, file_type, max_retries=3):
        for attempt in range(max_retries):
            try:
                await AssetImporter.import_file(file_path, file_type)
                return True
            except Exception as e:
                logging.warning(f"Import attempt {attempt + 1} failed: {str(e)}")
        return False


@unreal.uclass()
class AssetImporter(unreal.GlobalEditorUtilityBase):
    file_type_handler = FileTypeHandler()

    @staticmethod
    def import_with_interchange(file_path, file_type):
        # Create an Interchange import request
        import_request = unreal.InterchangeImportRequest()
        import_request.file_paths = [file_path]

        # Set import options based on file type
        import_options = unreal.InterchangeImportOptions()
        if file_type == FileType.GLTF:
            import_options.import_type = unreal.InterchangeImportType.STATIC_MESH
        elif file_type == FileType.SUBSTANCE:
            import_options.import_type = unreal.InterchangeImportType.MATERIAL
        elif file_type == FileType.HOUDINI:
            import_options.import_type = unreal.InterchangeImportType.HOUDINI_ASSET

        import_request.import_options = import_options

        # Set destination path
        import_request.destination_path = f"/Game/ImportedAssets/{file_type.name}"

        # Create an Interchange manager and start the import
        manager = unreal.InterchangeManager.get_interchange_manager()
        result = manager.import_assets(import_request)

        if result.import_succeeded():
            logging.info(f"Successfully imported: {file_path}")
            return True
        else:
            logging.error(f"Failed to import: {file_path}")
            for error in result.get_all_results():
                logging.error(f"Error: {error.get_message()}")
            return False

    @staticmethod
    async def import_file(file_path, file_type):
        logging.info(f"Starting import of {file_path}")

        try:
            if AssetImporter.import_with_interchange(file_path, file_type):
                logging.info(f"Successfully imported {file_path}")
            else:
                raise Exception("Import failed")
        except Exception as e:
            logging.error(f"Error importing {file_path}: {str(e)}")
            raise


class FileWatcher:
    def __init__(self, directory):
        self.directory = directory
        self.processed_files = set()
        self.file_queue = asyncio.Queue()
        self.worker_pool = ImportWorkerPool()

    async def watch_directory(self):
        while True:
            for root, _, files in os.walk(self.directory):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    if full_path not in self.processed_files:
                        await self.file_queue.put(full_path)
                        self.processed_files.add(full_path)
            await asyncio.sleep(1)

    async def process_files(self):
        while True:
            file_path = await self.file_queue.get()
            file_extension = os.path.splitext(file_path)[1].lower()
            file_type = AssetImporter.file_type_handler.get_file_type(file_extension)

            if file_type is None:
                logging.warning(f"Unsupported file type: {file_extension}")
                continue

            if not await ErrorHandler.retry_import(file_path, file_type):
                logging.error(f"Import failed after max retries: {file_path}")

            self.file_queue.task_done()

    async def run(self):
        await asyncio.gather(
            self.watch_directory(),
            self.process_files()
        )


def start_file_watcher(directory):
    watcher = FileWatcher(directory)
    asyncio.run(watcher.run())


# Example usage
if __name__ == "__main__":
    watch_directory = "C:/Path/To/Export/Directory"
    start_file_watcher(watch_directory)