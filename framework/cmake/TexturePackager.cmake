
#
# Texture Packager
# Convert PNG textures from the specified path into its media equivalent.
#

function(add_textures_from_path _path)
    cmake_parse_arguments(args "UASTC" "SCALE;DESTINATION" "" ${ARGN})

    if(DEFINED GLOBAL_ASSET_BASE_PATH)
        set(_path "${GLOBAL_ASSET_BASE_PATH}/${_path}")
    endif() 

    if(NOT DEFINED args_SCALE)
        set(args_SCALE 1.0)
    endif()

    if(DEFINED args_UASTC)
        set(TEXTURE_COMPRESSION "--encode" "uastc" "--zcmp")
    else()
        set(TEXTURE_COMPRESSION "")
    endif()

    set(CONVERTER_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/../../project/tools/toktx.exe")
    if(NOT EXISTS ${CONVERTER_TOOL})
        message(WARNING "TexturePackager -> Texture converter tool wasn't found: ${CONVERTER_TOOL}")
        return()
    endif()

    set(TEXTURE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Media/Textures")
    if(DEFINED args_DESTINATION)
        set(TEXTURE_PATH "${args_DESTINATION}")
    elseif(DEFINED TEXTURE_DESTINATION)
        set(TEXTURE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${TEXTURE_DESTINATION}")
    endif()

    if(NOT EXISTS ${TEXTURE_PATH})
        file(MAKE_DIRECTORY ${TEXTURE_PATH})
    endif()

    set(ASSET_CACHE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../build/assets_cache/textures")
    if(NOT EXISTS ${ASSET_CACHE_PATH})
        file(MAKE_DIRECTORY ${ASSET_CACHE_PATH})
    endif()

    file(GLOB png_textures "${_path}/*.png")
    list(LENGTH png_textures total_textures)

    message(STATUS "Preparing to convert ${total_textures} textures from '${_path}'")
    
    set(current_index 0)
    foreach(file ${png_textures})
        math(EXPR current_index "${current_index} + 1")
        get_filename_component(output_filename ${file} NAME_WE)
        set(cached_output "${ASSET_CACHE_PATH}/${output_filename}.ktx")
        set(final_output "${TEXTURE_PATH}/${output_filename}.ktx")

        if(EXISTS ${cached_output})
            file(COPY ${cached_output} DESTINATION ${TEXTURE_PATH})
        else()
            set(PARAMS
                "--genmipmap"
                --scale ${args_SCALE}
                ${TEXTURE_COMPRESSION}
                "--"
                "${cached_output}"
                "${file}"
            )
            execute_process(COMMAND "${CONVERTER_TOOL}" ${PARAMS}
                            ERROR_VARIABLE conv_error
                            RESULT_VARIABLE conv_retval)

            if(conv_error)
                message(WARNING "TexturePackager -> ${conv_error}")
            endif()

            file(COPY ${cached_output} DESTINATION ${TEXTURE_PATH})
        endif()

        # Progress bar simulation
        math(EXPR percent_done "(${current_index} * 100) / ${total_textures}")
        math(EXPR filled "(${percent_done} / 10)")
        math(EXPR empty "10 - ${filled}")

        set(bar "")
        if(filled GREATER 0)
            foreach(i RANGE 1 ${filled})
                set(bar "${bar}#")
            endforeach()
        endif()

        if(empty GREATER 0)
            foreach(i RANGE 1 ${empty})
                set(bar "${bar}-")
            endforeach()
        endif()

        message(STATUS "[${bar}] ${percent_done}% - Processed: '${output_filename}'")

    endforeach()
endfunction()

function(add_texture _texture_path)
    cmake_parse_arguments(args "UASTC" "SCALE;DESTINATION" "" ${ARGN})

    if(DEFINED GLOBAL_ASSET_BASE_PATH)
        set(_texture_path "${GLOBAL_ASSET_BASE_PATH}/${_texture_path}")
    endif()  

    if(NOT EXISTS ${_texture_path})
        message(WARNING "TexturePackager -> Requested texture doesn't exist: ${_texture_path}")
        return()
    endif()

    if(NOT DEFINED args_SCALE)
        set(args_SCALE 1.0)
    endif()

    if(DEFINED args_UASTC)
        set(TEXTURE_COMPRESSION "--encode" "uastc" "--zcmp")
    else()
        set(TEXTURE_COMPRESSION "")
    endif()    

    set(TEXTURE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Media/Textures")
    if(DEFINED args_DESTINATION)
        set(TEXTURE_PATH "${args_DESTINATION}")
    elseif(DEFINED TEXTURE_DESTINATION)
        set(TEXTURE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${TEXTURE_DESTINATION}")
    endif()

    set(ASSET_CACHE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../build/assets_cache/textures")
    if(NOT EXISTS ${ASSET_CACHE_PATH})
        file(MAKE_DIRECTORY ${ASSET_CACHE_PATH})
    endif()

    if(NOT EXISTS ${TEXTURE_PATH})
        file(MAKE_DIRECTORY ${TEXTURE_PATH})
    endif()

    get_filename_component(output_filename ${_texture_path} NAME_WE)
    get_filename_component(output_ext ${_texture_path} EXT)

    set(dst_output_path "${TEXTURE_PATH}/${output_filename}.ktx")
    set(cached_output "${ASSET_CACHE_PATH}/${output_filename}.ktx")

    if(EXISTS ${dst_output_path})
        return()
    endif()

    if(output_ext STREQUAL ".ktx")
        message("Copying KTX texture: '${output_filename}'")
        file(COPY ${_texture_path} DESTINATION ${TEXTURE_PATH})
        return()
    endif()

    set(CONVERTER_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/../../project/tools/toktx.exe")
    if(NOT EXISTS ${CONVERTER_TOOL})
        message(WARNING "TexturePackager -> Texture converter tool wasn't found: ${CONVERTER_TOOL}")
        return()
    endif()

    message("Converting texture: '${output_filename}' to KTX")
    set(PARAMS
        "--genmipmap"
        --scale ${args_SCALE}
        ${TEXTURE_COMPRESSION}
        "--"
        "${cached_output}"
        "${_texture_path}"
    )
    execute_process(COMMAND "${CONVERTER_TOOL}" ${PARAMS}
                    ERROR_VARIABLE conv_error
                    RESULT_VARIABLE conv_retval)

    if(conv_error)
        message(WARNING "TexturePackager -> ${conv_error}")
    endif()

    file(COPY ${cached_output} DESTINATION ${TEXTURE_PATH})
endfunction()

function(add_all_ktx_from_path _path)
    cmake_parse_arguments(args "UASTC" "SCALE;DESTINATION" "" ${ARGN})

        if(DEFINED GLOBAL_ASSET_BASE_PATH)
        set(_path "${GLOBAL_ASSET_BASE_PATH}/${_path}")
    endif() 

    if(DEFINED args_SCALE)
        message(DEBUG "WARNING: Scale argument passed to add_all_ktx_from_path() which is not implemented, ignoring argument.")
    endif()

    if(DEFINED args_UASTC)
        message(DEBUG "WARNING: UASTC argument passed to add_all_ktx_from_path() which is not implemented, ignoring argument.")
    endif()

    set(TEXTURE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Media/Textures")
    if(DEFINED args_DESTINATION)
        set(TEXTURE_PATH "${args_DESTINATION}")
    elseif(DEFINED TEXTURE_DESTINATION)
        set(TEXTURE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${TEXTURE_DESTINATION}")
    endif()

       if(NOT EXISTS ${TEXTURE_PATH})
        file(MAKE_DIRECTORY ${TEXTURE_PATH})
    endif()

    file(GLOB ktx_textures "${_path}/*.ktx")
    list(LENGTH ktx_textures total_textures)
    
    message(STATUS "Preparing to copy ${total_textures} textures from '${_path}'")
    
    foreach(file ${ktx_textures})
        math(EXPR current_index "${current_index} + 1")
    
        get_filename_component(output_filename ${file} NAME)
        set(final_output "${TEXTURE_PATH}/${output_filename}")
        file(COPY "${_path}/${output_filename}" DESTINATION "${TEXTURE_PATH}")

        # Progress bar simulation
        math(EXPR percent_done "(${current_index} * 100) / ${total_textures}")
        math(EXPR filled "(${percent_done} / 10)")
        math(EXPR empty "10 - ${filled}")

        set(bar "")
        if(filled GREATER 0)
            foreach(i RANGE 1 ${filled})
                set(bar "${bar}#")
            endforeach()
        endif()

        if(empty GREATER 0)
            foreach(i RANGE 1 ${empty})
                set(bar "${bar}-")
            endforeach()
        endif()

        message(STATUS "[${bar}] ${percent_done}% - Processed: '${output_filename}'")

    endforeach()
endfunction()

function(add_ktx _texture_path)

    if(DEFINED GLOBAL_ASSET_BASE_PATH)
        set(_texture_path "${GLOBAL_ASSET_BASE_PATH}/${_texture_path}")
    endif()  

    if(NOT EXISTS ${_texture_path})
        message(WARNING "TexturePackager -> Requested texture doesn't exist: ${_texture_path}")
        return()
    endif()  

    set(TEXTURE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Media/Textures")
    if(DEFINED args_DESTINATION)
        set(TEXTURE_PATH "${args_DESTINATION}")
    elseif(DEFINED TEXTURE_DESTINATION)
        set(TEXTURE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${TEXTURE_DESTINATION}")
    endif()

    if(NOT EXISTS ${TEXTURE_PATH})
        file(MAKE_DIRECTORY ${TEXTURE_PATH})
    endif()

    if(EXISTS "${TEXTURE_PATH}/${_texture_path}")
        return()
    endif()

    message(STATUS "Copying KTX texture: '${output_filename}'")
    file(COPY ${_texture_path} DESTINATION ${TEXTURE_PATH})

endfunction()