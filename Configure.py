#
# Copyright (c) 2025 QUALCOMM Technologies Inc.
# All Rights Reserved.
#
import sys
import os
import argparse
import hashlib
import pathlib
import platform
import shutil
import tarfile
import subprocess
import tempfile
import zipfile

tab_size = 8                                # hard tab size

parser = argparse.ArgumentParser(
            description="Configuration tool for Snapdragon Studios's Sample Framework",
            epilog="see README.md for additional help")
parser.add_argument('--build', help = 'Build with the current ConfigLocal settings, does require user input.  Typically this option is used for subsequent builds after initial configuration.', action = 'store_true')
parser.add_argument('--clean', help = 'Clean all targets', action = 'store_true')
arguments = parser.parse_args()
enableUserInterface = not arguments.clean and not arguments.build


class Item:
    def __init__(self, name, selected = -1, readOnly = False, saveable = True, groups = [], dependencies = [], location = None, function = None):
        self.name = name
        self.groups = groups
        self.dependencies = dependencies
        self.selected = selected
        self.readOnly = readOnly
        self.saveable = saveable
        self.open = False
        self.location = location
        self.function = function
    def select(self):
        if self.function:
            self.function(self)
        else:
            toggle_selected(self)

class DisplayLine:
    def __init__(self, item: Item, string: str):
        self.item = item
        self.string = string
    def open(self):
        self.item.open = True
    def close(self):
        self.item.open = False

# determine correct keyboard handling code (per platform - we jump through these hoops so we do not need to install any external python libraries such as curses or pygame)
try:
    import msvcrt
    def get_key_blocking_windows():
        key = msvcrt.getch()
        keyCode = 0
        if key ==  b'\xe0' or key == b'\000':
            keyCode = int.from_bytes(msvcrt.getch(),byteorder='little')
            return ('c' + str(keyCode))             # turns into c?? (where ?? is the decimal representation of the keyCode)
        else:
            return str(key, encoding='ascii')
    get_key_blocking = get_key_blocking_windows
except ImportError:
    import tty, sys, termios    
    def get_key_blocking_unix():
        stdinFd = sys.stdin.fileno()
        savedAttr = termios.tcgetattr(stdinFd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            # read another character if escaped (special key sequence such as cursor)
            if key[0] == '\x1b':
                key = 'e'
                while True:
                    keyEscaped = sys.stdin.read(1)
                    key += keyEscaped
                    if keyEscaped.isalpha():
                        break
        finally:
            termios.tcsetattr(stdinFd, termios.TCSADRAIN, savedAttr)
        return key
    get_key_blocking = get_key_blocking_unix    


def toggle_selected(item: Item):
    # toggle selection status of the item
    recalculate_selections( groups, {item : 1 if item.selected == -1 else -1} )

def count_indent(s):
    indent = 0
    for c in s:
        if c.isspace():
            indent += 1
        elif c == '\t':
            indent = ((indent + tab_size) / tab_size) * tab_size
        else:
            break
    return indent

def read_config_group(configLines, currentIndent = 0):
    itemStrings = []
    groups = {}
    currentLineIdx = 0
    while currentLineIdx < len(configLines):
        configLine = configLines[currentLineIdx].rstrip()
        #print(configLine + " " + str(currentIndent))

        indent = count_indent(configLine)
        configLine = configLine.strip()
        if len(configLine) == 0:
            # skip blank lines (or lines with no non-whitespace content)
            currentLineIdx += 1
            continue
        if currentLineIdx == 0 and indent != currentIndent:
            raise Exception("indent mismatch error")
        
        if indent > currentIndent:
            # start of child block
            numProcessedLines, groupItemStrings, groupItemGroups = read_config_group(configLines[currentLineIdx:], indent)
            currentLineIdx += numProcessedLines
            groups[itemStrings[-1][0]] = [groupItemStrings, groupItemGroups]
        elif indent < currentIndent:
            # end of our block
            break
        else:
            splitLine = [s.strip() for s in configLine.split(';', 1)]
            name = splitLine[0].replace('\\', '/')
            if len(splitLine) > 1 and len(splitLine[1]) > 0:
                dependencies = [os.path.normpath(x.strip().replace('\\', '/')) for x in splitLine[1].split(',')]
            else:
                dependencies = []
            #print(dependencies)
            itemStrings.append([name, dependencies])
            currentLineIdx += 1
    return currentLineIdx, itemStrings, groups
        

def read_config(folder, filename):
    fullFilename = os.path.join(folder, filename)
    if os.path.exists(fullFilename):
        with open(fullFilename, 'r') as file:
            configLines = file.readlines()
            currentIndent = 0
            _, itemStrings, itemGroups = read_config_group(configLines)
        #print(itemStrings)
        #print(itemGroups)

        # make the Items
        def make_items(itemStrings, itemGroups):
            items = []
            for i in itemStrings:
                name = i[0]
                dependencies = i[1]
                item_group = make_items(*itemGroups[name]) if name in itemGroups else []                

                splitLocation = [s.strip() for s in name.split("@")]
                name = splitLocation[0]
                location = splitLocation[1] if len(splitLocation)>1 else None
                items.append( Item(name, -1, False, True, item_group, dependencies, location) )
            return items
        items = make_items(itemStrings, itemGroups)

        #print(items)
        return items

# write a config file containing ONLY the items that are enabled (selected)
def write_enabled_config(groups, folder, filename):
    filename = os.path.join(folder, filename)
    with open(filename, 'w') as file:
        
        def write(groups, indent = 0):
            for entry in groups:
                if entry.saveable:
                    if entry.selected != -1:
                        #file.write(' ' * indent + entry.name + ((';' + ", ".join(entry.dependencies)) if len(entry.dependencies) > 0 else '') + '\n')
                        file.write(' ' * indent + entry.name + '\n')
                    write(entry.groups, indent + 1)
        write(groups)
    print("Wrote config to " + filename)


def recalculate_selections(groups, changed_selections):
    if not changed_selections:
        changed_selections = {i: i.selected for i in groups}
    new_selections = {}
    while True:
        for item in changed_selections:
            selected = changed_selections[item]
            item.selected = selected
            if item.groups and selected != 0:
                for i in item.groups:
                    new_selections[i] = selected
        if not new_selections:
            break
        changed_selections = new_selections
        new_selections = {}

        
    # now scan through and determine selection status (all, partial, none; 1 0 -1) for all items with children (from very top level down)
    def recalculate_parent_selections(group):
        allSelected = True
        allClear = True
        for item in group:
            if item.groups:
                childAllSelected, childAllClear = recalculate_parent_selections(item.groups)
                item.selected = -1 if childAllClear else 1 if childAllSelected else 0
                allSelected = allSelected and childAllSelected
                allClear = allClear and childAllClear
            else:
                allSelected = allSelected and item.selected == 1
                allClear = allClear and item.selected == -1
        if allClear and allSelected:
            raise Exception("logic error")
        return allSelected, allClear
    recalculate_parent_selections(groups)

    # and re-set all the dependencies!
    while True:
        def find_dependencies(groups):
            new_selections = {}
            for item in groups:
                if item.groups:
                     new_selections = {**new_selections, **find_dependencies(item.groups)}
                if item.selected != -1 and item.dependencies:
                    for dependencyName in item.dependencies:
                        dependency = items_by_path[dependencyName]
                        if dependency.selected != 1:
                            # add as a new dependeny to set
                            new_selections[dependency] = 1
            return new_selections
        new_selections = find_dependencies(groups)
        if not new_selections:
            break
        # set all the dependencies
        recalculate_selections(groups, new_selections)

def clean_items(itemsToClean):
    for item in itemsToClean:
        cleanFunctionsMap[item.name]()

def build_items(itemsToBuild):
    success = True
    for item in itemsToBuild:
        if not buildFunctionsMap[item.name]():
            success = False
    return success

def save_and_begin_processing_selected(_: Item):
    write_enabled_config(groups, ".", "ConfigLocal.txt")
    write_cmake_config(projectsItems, "ConfigLocal.cmake")
    write_gradle_properties(projectsItems, "ConfigLocal.properties")
    write_android_gradle_files(projectsItems)

    # clean selected items
    itemsToClean = filter( lambda x: x.selected == 1, cleanTargetItems )
    clean_items( itemsToClean )

    # download selected items
    download(projectsItems)

    # see how many projects are set to build
    selected_project_count = count_selected(projectsItems)
    if selected_project_count == 0:
        print("\nNo projects selected to be built")
    else:
        # build selected items
        targetsToBuild = list(filter( lambda x: x.selected == 1, buildTargetItems ))
        if not targetsToBuild:
            print("\nNo targets selected to be built")
        else:
            if not build_items( targetsToBuild ):
                print("ERROR BUILDING TARGETS (see output above)")

    print("\nCompleted.")
    # quit the app!
    exit(0)

class Download:
    def __init__(self, outputFolder, url, downloadDestination, md5Expected, patchFile):
        self.outputFolder = outputFolder
        self.url = url
        self.downloadDestination = downloadDestination
        self.md5Expected = md5Expected
        self.patchFile = patchFile

def find_downloads(groups, folder, onlySelected):
    downloads = []
    for item in groups:
        if not onlySelected or item.selected != -1:
            if item.groups:
                if item.name[-1] == '/':
                    subFolder = os.path.join(folder, item.name[:-1])
                else:
                    subFolder = folder
                downloads.extend( find_downloads(item.groups, subFolder, onlySelected) )
            if item.location:
                md5Expected = None
                patchFile = None
                security = item.location.split(' ', 2)
                if len(security) > 1:
                    for s in security[1:]:
                        if s[:4] == 'MD5:':
                            md5Expected = s[4:]
                        elif s[:6] == 'PATCH:':
                            patchFile = os.path.normpath(os.path.join(folder, os.path.normcase(s[6:])))
                        url = security[0]
                elif len(security) == 1:
                    url = security[0]
                else:
                    raise Exception("Error parsing download string " + item.location + ".  Format is <url> [ MD5:<md5>] [ PATCH:<patchfilename>]")
                                
                outputFolder = os.path.join(folder, item.name)
                downloadDestination = os.path.join(folder, url.rsplit('/', 1)[-1])
                downloads.append( Download(outputFolder, url, downloadDestination, md5Expected, patchFile) )
    return downloads

def download(groups):
    
    downloads = find_downloads(groups, '.', onlySelected=True)

    if not downloads:
        return False
    print("Preparing to download\n---------------------")
    for download in downloads:
        print("" + download.url + "\n  ->  " + download.outputFolder)
    print("\n")

    for download in downloads:
        outputFolder = download.outputFolder
        url = download.url
        downloadDestination = download.downloadDestination
        md5Expected = download.md5Expected
        patchFile = download.patchFile

        refreshFiles = False    # assume everything up to date
        
        if os.path.exists(downloadDestination):
            print("Found " + downloadDestination + " (skipping download)")
            filename = downloadDestination
        else:
            destinationFolder = os.path.dirname(downloadDestination)
            if not os.path.exists(destinationFolder):
                print("Creating folder " + destinationFolder)
                pathlib.Path(destinationFolder).mkdir(parents = True, exist_ok = True)

            print("Downloading " + url)
            refreshFiles = True
            import urllib.request, ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            filename, headers = urllib.request.urlretrieve(url)

            if md5Expected:
                md5hash = hashlib.md5()
                with open(filename, 'rb') as file:
                    for b in iter(lambda: file.read(32768), b""):
                        md5hash.update(b)
                if md5hash.hexdigest() != md5Expected:
                    os.remove(filename)
                    raise Exception("MD5 hash security check for " + filename + " (" + downloadDestination + ") failed.\n  Expected " + md5Expected + " got " + md5hash.hexdigest())
                print("  MD5 hash verified")

            shutil.move(filename, downloadDestination)
            filename = downloadDestination
            print("  copied to " + filename)
        
        # prepare for files copied over destination folder
        if refreshFiles and os.path.exists(outputFolder):
            shutil.rmtree(outputFolder)

        if not os.path.exists(outputFolder):
            filenameExtSplit = filename.rsplit('.', 2)
            with tempfile.TemporaryDirectory() as tempExtractDir:   # use a temporary directory as an extraction location
                if len(filenameExtSplit)>1 and filenameExtSplit[-1].lower() == 'zip':
                    print("  extracting zip to " + tempExtractDir)
                    with zipfile.ZipFile(filename, 'r') as file:
                        file.extractall(tempExtractDir)
                elif len(filenameExtSplit)>1 and filenameExtSplit[-2].lower() == 'tar':
                    print("  extracting tar to " + tempExtractDir)

                    with tarfile.open(filename) as tar:
                        # check the tar only contains files/directories (no symlinks etc supported) and no attempts to write outside of the target directory
                        targetDir = pathlib.Path(outputFolder)
                        for tarinfo in tar.getmembers():
                            if tarinfo.isfile() or tarinfo.isdir():
                                tarinfo.uname = None
                                tarinfo.gname = None
                                outputPath = pathlib.Path( os.path.normpath(os.path.join(targetDir, tarinfo.name)) )
                                if targetDir not in outputPath.parents:
                                    raise Exception("Tar file " + filename + " prevented from write outside of safe directory structure (potential security issue).  (" + str(outputPath) + ")")

                        if sys.version_info[0] == 3 and sys.version_info[1] >= 12:
                            tar.extractall(path=tempExtractDir, filter = 'data')
                        else:
                            tar.extractall(path=tempExtractDir)
                else:
                    raise Exception("Unknown file archive format for " + filename + " ( .tar.gz / .tar.xz / .zip supported)")

                # all the achives have a top level directory (and the contents we expect below it), get that directory name
                print("  moving extracted files to " + outputFolder)
                tempExtractDirContents = os.listdir(tempExtractDir)
                if len(tempExtractDirContents)!=1:
                    raise Exception("Expected downloaded archive from " + url + " to contain a single top level folder, does not - actual contents: " + tempExtractDirContents)
                extractedFolderPath = os.path.join(tempExtractDir, tempExtractDirContents[0])

                # Move the top level folder out of the temporary extraction location to the correctly named destination
                shutil.copytree(extractedFolderPath, outputFolder)

                if patchFile:
                    print("  patching using " + patchFile)
                    subprocess.run(["git", "apply","--verbose","--unsafe-paths","--ignore-space-change","--directory", outputFolder, patchFile])

        elif refreshFiles:
            raise Exception("Was expecting folder to be removed before extraction, try manually deleting?  (folder: " + outputFolder + ")")
        else:
            print("  skipped extracting files to " + outputFolder + " (folder already exists)")
    return True # did something

# clean anything that (may have been) downloaded
def clean_externals():
    downloads = find_downloads(groups, '.', onlySelected=False)
    for d in downloads:
        if os.path.exists(d.downloadDestination):
            print("Removing: " + d.downloadDestination)
            os.remove(d.downloadDestination)    
        if os.path.isdir(d.outputFolder):
            print("Removing: " + d.outputFolder)
            shutil.rmtree(d.outputFolder, ignore_errors=True)

def write_cmake_config(groups, filename):
    with open(filename, "wt") as file:
        file.write("# CMake file written by " + sys.argv[0] + "\n#\n# Defines which framework libraries and samples are enabled\n# Do not check this file in to version control (generate locally)\n\n")
        file.write("cmake_minimum_required (VERSION 3.21)\n\n")
        def write_items(groups, parentName, file):
            for item in groups:
                variableName = parentName + "_" + item.name.strip('\\/')
                file.write("SET(" + variableName + (" True)\n" if item.selected != -1 else " False)\n"))
                if item.groups:
                    if item.name[-1] == '/':
                        write_items(item.groups, variableName, file)
                    else:
                        write_items(item.groups, parentName, file)
        write_items(groups, "FRAMEWORK", file)
        print("Cmake build properties file \"" +  filename + "\" written")
    
def write_gradle_properties(groups, filename):
    with open(filename, "wt") as file:
        file.write("# Gradle properties file written by " + sys.argv[0] + "\n#\n# Defines which framework libraries and samples are enabled\n# Do not check this file in to version control (generate locally)\n\n")
        def write_items(groups, parentName, file):
            for item in groups:
                variableName = parentName + "_" + item.name.strip('\\/')
                file.write(variableName + (" = true\n" if item.selected != -1 else " = false\n"))
                if item.groups:
                    if item.name[-1] == '/':
                        write_items(item.groups, variableName, file)
                    else:
                        write_items(item.groups, parentName, file)
        write_items(groups, "FRAMEWORK", file)
        print("Gradle (Android) build properties file \"" + filename + "\" written")



def write_android_gradle_files(groups):
    """
    * Generates Android build.gradle files for selected sample and test projects.
    * @param groups : The top-level configuration groups containing all selectable items.
    * @note This function creates build.gradle files under ./samples/<project>/build or ./tests/<project>/build
    """
    build_gradle_template = """apply plugin: 'com.android.application'

android {{
    compileSdkVersion 33
    namespace "com.quic.{project_name}"
    lintOptions {{
        abortOnError false
    }}

    String rootDir = "${{project.rootDir}}"
    rootDir = rootDir.replace("\\\\", "/")

    defaultConfig {{
        applicationId "com.quic.{project_name}"
        minSdkVersion 26
        targetSdkVersion 33
        versionCode 1
        versionName "1.0"
        ndkVersion "${{project.ndkVersionDefault}}"
        ndk {{
            abiFilters 'arm64-v8a'
        }}
        externalNativeBuild {{
            cmake {{
                arguments "-DPROJECT_ROOT_DIR=${{rootDir}}", "-DFRAMEWORK_DIR=${{rootDir}}/../../framework"
            }}
        }}
     }}

    signingConfigs{{
        unsigned{{
            storeFile file("${{System.env.USERPROFILE}}/.android/debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
            v2SigningEnabled = false
        }}
    }}

    buildTypes {{
        release {{
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
            signingConfig signingConfigs.unsigned
        }}
        debug {{
            debuggable = true
            jniDebuggable = true
        }}
    }}

    sourceSets {{
        main {{
            jni.srcDirs = []
            manifest.srcFile '../project/android/AndroidManifest.xml'
            res.srcDirs = ['../project/android/res']
        }}
        main.assets {{
            srcDirs = ['assets']
            srcDirs += ['assets_tmp']
        }}
    }}

    dependencies {{
        implementation project(':framework')
    }}

    externalNativeBuild {{
        cmake {{
            version "3.25.0+"
            path "../CMakeLists.txt"
            buildStagingDirectory "../../../build/cxx/${{project.name}}"
        }}
    }}

    task copyTmpAssets(type: Copy) {{
        from "Media"
        into "assets_tmp/build/Media"
    }}
    task removeTmpAssets(type: Delete) {{
        delete "assets_tmp"
    }}

    afterEvaluate {{
        packageRelease.finalizedBy(removeTmpAssets)
        preBuild.dependsOn(":framework:buildNeeded")

        // Fix: CMake writes assets (shaders, meshes, textures) to build/Media/ during configure+build,
        // which runs AFTER preBuild. Running copyTmpAssets at preBuild means the source is empty on
        // first build. Fix: run copyTmpAssets after native build, before asset merge.
        def nativeBuildDebug   = tasks.findByName("externalNativeBuildDebug")
        def nativeBuildRelease = tasks.findByName("externalNativeBuildRelease")
        def mergeDebugAssets   = tasks.findByName("mergeDebugAssets")
        def mergeReleaseAssets = tasks.findByName("mergeReleaseAssets")
        if (nativeBuildDebug)   copyTmpAssets.dependsOn(nativeBuildDebug)
        if (nativeBuildRelease) copyTmpAssets.dependsOn(nativeBuildRelease)
        if (mergeDebugAssets)   mergeDebugAssets.dependsOn(copyTmpAssets)
        if (mergeReleaseAssets) mergeReleaseAssets.dependsOn(copyTmpAssets)
    }}

    def overrideFile = file("${{project.projectDir}}/../override.gradle")
    if (overrideFile.exists()) {{
        apply from: overrideFile
    }}
}}
"""

    def write_items(groups):
        for item in groups:
            if item.selected != -1 and not item.groups:
                # Check for existence in samples and tests
                possible_paths = [os.path.join("samples", item.name), os.path.join("tests", item.name)]
                target_path = None
                for path in possible_paths:
                    if os.path.isdir(path):
                        target_path = path
                        break

                if target_path:
                    build_dir = os.path.join(target_path, "build")
                    os.makedirs(build_dir, exist_ok=True)
                    gradle_path = os.path.join(build_dir, "build.gradle")
                    with open(gradle_path, "w") as f:
                        f.write(build_gradle_template.format(project_name=item.name))
                    print(f"Generated build.gradle for {target_path}")
                else:
                    print(f"Skipped {item.name}: no matching folder in samples/ or tests/")
            
            if item.groups:
                write_items(item.groups)

    write_items(groups)

def count_selected(groups):
    count = 0
    for item in groups:
        if item.groups:
            count += count_selected(item.groups)
        else:
            count += 1 if item.selected != -1 else 0
    return count
    

# load the list of projects (that we can build) and their dependencies, from Config.txt
projectsItems = read_config(".", "Config.txt")

#
# setup top level menu items
#

# Clean functions
cleanFunctionsMap = {
    "Android"      : lambda: shutil.rmtree("./build",                         ignore_errors=True),
    "Linux"        : lambda: shutil.rmtree("./project/linux/solution",        ignore_errors=True),
    "Windows ARM64": lambda: shutil.rmtree("./project/windows/solutionArm64", ignore_errors=True),
    "Windows x64"  : lambda: shutil.rmtree("./project/windows/solution",      ignore_errors=True),
    "Externals"    : clean_externals
}

# run the given batch file and return True if successful (errorcode == 0)
def RunBatch(filename):
    retcode = subprocess.call([os.path.abspath(filename)], shell=False)
    return retcode == 0

cleanTargetItems = [Item(n) for n in cleanFunctionsMap]
# Build functions
if platform.system() == 'Windows':
    buildFunctionsMap = {
        "Tools"        : lambda: RunBatch("./project/tools/build.bat"),
        "Android"      : lambda: RunBatch("./project/android/build.bat"),
        "Windows ARM64": lambda: RunBatch("./project/windows/buildArm64.bat"),
        "Windows x64"  : lambda: RunBatch("./project/windows/build.bat"),
    }
else:
    buildFunctionsMap = {
        "Tools"        : lambda: subprocess.Popen('source ' + os.path.abspath("./project/tools/build.sh"), shell=True, executable="/bin/bash"),
        "Linux"        : lambda: subprocess.Popen('source ' + os.path.abspath("./project/linux/build.sh"), shell=True, executable="/bin/bash"),
    }
buildTargetItems = [Item(n) for n in buildFunctionsMap]    

groups = [
    Item("Clean",                     selected = False, saveable = False, groups = cleanTargetItems),
    Item("Build Targets",             selected = False, groups = buildTargetItems),
    Item("Build Projects",            groups = projectsItems),
    Item("Save And Begin Processing", function = save_and_begin_processing_selected)
]

# for all the projects build a lookup by 'path' (for named dependencies)
def make_items_by_path(groups, path):
    items_by_path = {}
    for item in groups:
        if item.groups:
            # not a leaf item
            if item.name[-1] == '/':
                # folder based grouping
                group_items_by_path = make_items_by_path(item.groups, os.path.join(path, item.name[:-1]))
            else:
                # non folder grouping
                group_items_by_path = make_items_by_path(item.groups, path)
            if group_items_by_path:
                items_by_path = {**items_by_path, **group_items_by_path}
        else:
            # leaf node (or empty 'folder')
            if item.name[-1] == '/':
                pass # we dont need to record empty folders!
            else:
                items_by_path[os.path.join(path, item.name)] = item
    return items_by_path
items_by_path = make_items_by_path(groups, "")


# now load the 'local' settings (which items/projects are currently selected for our local build)
if groups:
    localSelectedGroups = read_config(".", "ConfigLocal.txt")
    if not localSelectedGroups:
        localSelectedGroups = []

    def set_selected(items, override_items):
        leaf_selections = {}
        items_dict = { g.name: g for g in items }
        for override in override_items:
            if override.name in items_dict:
                item = items_dict[override.name]
                if override.groups:
                    # parent node
                    leaf_selections = {**leaf_selections, **set_selected( item.groups, override.groups )}
                else:
                    # leaf node
                    leaf_selections[item] = 1
        return leaf_selections
    recalculate_selections(groups, set_selected(groups, localSelectedGroups))

# recaulculate whether items are selected, unselected or partially selected (some children selected and some unselected)
recalculate_selections(groups, {})


# see if we are running the user interface or in 'commandline' mode!
if enableUserInterface:

    #
    # Gui Mode
    #

    # open the top level project groups
    for topLevelItem in projectsItems:
        topLevelItem.open = True


    quitMode = 0
    cursorIdx = 0
    topLine = 0

    # get number of lines in scrolling window
    numLines = os.get_terminal_size().lines - 5
    # clear terminal
    print('\x1bc')

    # UI loop (draw menu and wait-for / react-to key input)
    while(True):
        indent = 0
        
        def make_display(item : Item, cursorIdx, indent = 0):
            cursor = '>>' if cursorIdx == 0 else '  '
            selected = ''
            if not item.function:
                selected = '[-]'
                if item.selected == 1:
                    selected = '[X]'
                elif item.selected == -1:
                    selected = '[ ]'
            more = '' if not item.groups else '...'
            display = [DisplayLine(item, str(cursor + ' ' * indent + selected + item.name + more))]
            if item.open:
                for group in item.groups:
                    display.extend(make_display(group, cursorIdx - len(display), indent + 2))
            return display

        all_display = []
        for group in groups:
            all_display.extend( make_display(group, cursorIdx - len(all_display)) )

        for d in all_display[topLine : topLine+numLines]:
            print(d.string)
        for r in range(0, numLines - len(all_display)):
            print() 
        print()
        if quitMode == 0:
            if all_display[cursorIdx].item.function:
                print('Up/Down - Move, Return = Select')
            else:
                print('Up/Down - Move, Return = Toggle selection' + (', Right/left - Open/close sub-menu' if all_display[cursorIdx].item.groups else ''))
            print('Q = Quit, S - Save Settings')
        elif quitMode == 1:
            print('B - Back, W - Write settings and quit, Q - Quit without saving')
        
        key = get_key_blocking() #msvcrt.getch()
        print('\x1bc')
        if quitMode ==0:
            if key == 'x' or key == 'X':
                quitMode = 1
            elif key == 'q' or key == 'Q':
                quitMode = 1
            elif key == 's' or key == 'S':
                write_enabled_config(groups, ".", "ConfigLocal.txt")
            elif key == ' ' or key == '\r':
                # select item (call function)
                item = all_display[cursorIdx].item
                item.select()
            elif key == 'c72' or key == 'e[A': # up
                if cursorIdx == 0:
                    cursorIdx = len(all_display) - 1
                    topLine = max(0, len(all_display) - numLines)
                else:
                    cursorIdx = cursorIdx - 1
                    if topLine >= cursorIdx - 1 and topLine > 0:
                        topLine = max(0, cursorIdx - 1)
            elif key == 'c80' or key == 'e[B': # down
                cursorIdx = cursorIdx + 1
                if cursorIdx == len(all_display):
                    cursorIdx = 0
                    topLine = 0
                else:
                    if cursorIdx >= topLine + numLines - 2 and topLine + numLines < len(all_display):
                        topLine = max(0, min(cursorIdx - numLines + 2, len(all_display) - numLines))
            elif key == 'c77' or key == 'e[C': # right
                all_display[cursorIdx].open()
            elif key == 'c75' or key == 'e[D': # left
                all_display[cursorIdx].close()

            if topLine >= cursorIdx - 1 and topLine > 0:
                topLine = cursorIdx - 1

        elif quitMode == 1:
            if key == 'b' or key == 'B':
                quitMode = 0
            elif key == 'w' or key == 'W':
                # for g in groups:
                #   write_folder(g)
                write_enabled_config(groups, ".", "ConfigLocal.txt")
                exit(0)
            elif key == 'q' or key == 'Q':
                print("Exited without writing")
                exit(0)

else:
    #
    # Commandline mode
    #
    write_cmake_config(projectsItems, "ConfigLocal.cmake")
    write_gradle_properties(projectsItems, "ConfigLocal.properties")
    write_android_gradle_files(projectsItems)

    if arguments.clean:
        # clean everything!
        clean_items( cleanTargetItems )

    if arguments.build:
        # always do the downloads (build will fail if we are missing a download we are dependent on)
        download(projectsItems)

        # only build what was previously selected (and saved)
        targetsToBuild = list(filter( lambda x: x.selected == 1, buildTargetItems))
        if not targetsToBuild:
            print("\nNo targets to build (did you run 'python Configure.py' and select something to build?)")
        else:
            selected_project_count = count_selected(projectsItems)
            if selected_project_count == 0:
                print("\nNo projects selected to be built (did you run 'python Configure.py' and select something to build?)")
            else:
                build_items( targetsToBuild )
