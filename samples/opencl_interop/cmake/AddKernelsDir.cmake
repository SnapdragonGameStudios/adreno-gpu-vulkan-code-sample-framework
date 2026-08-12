#
# Copy OpenCL kernels
# Add everything with .cl extension from the kernels/ directory.
#

function(copy_kernels files dst_dir target_name)
    set(output_files "")
    foreach(file ${files})
        get_filename_component(output_filename ${file} NAME)
        set(output_kernel "${dst_dir}/${output_filename}")

        add_custom_command(
            OUTPUT ${output_kernel}
            MAIN_DEPENDENCY ${file}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${dst_dir}
            COMMAND ${CMAKE_COMMAND} -E copy ${file} ${output_kernel}
        )
        list(APPEND output_files ${output_kernel})
    endforeach()

    add_custom_target(${target_name} ALL DEPENDS ${output_files})
    set_target_properties(${target_name} PROPERTIES FOLDER "kernels")
    if(TARGET ${PROJECT_NAME})
        add_dependencies(${PROJECT_NAME} ${target_name})
    elseif(DEFINED TARGET_NAME AND TARGET ${TARGET_NAME})
        add_dependencies(${TARGET_NAME} ${target_name})
    endif()
endfunction()

function(scan_for_kernels)
    # Optional destination path for copied kernels
    set(KERNEL_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Media/Kernels")
    if(DEFINED KERNEL_DESTINATION)
        set(KERNEL_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_DESTINATION}")
    endif()

    # Use project name to generate unique target names
    set(target_prefix "${PROJECT_NAME}")

    # Scan through kernels directory looking for OpenCL kernel source files and copy them
    file(GLOB kernel_files "kernels/*.cl")
    copy_kernels("${kernel_files}" "${KERNEL_OUTPUT_PATH}" "${target_prefix}_KERNELS")

    # Add kernels (sources) into a 'kernels' folder for Visual Studio
    source_group("kernels" FILES ${kernel_files})
endfunction()