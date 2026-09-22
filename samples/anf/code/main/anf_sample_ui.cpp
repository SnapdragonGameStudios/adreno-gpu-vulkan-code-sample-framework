// Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "anf_sample_ui.hpp"

#include "anf_gpu_profiler.hpp"
#include "imgui.h"

#include <algorithm>

AnfSampleUiResult DrawAnfSampleUi(const AnfSampleUiModel& model, const AnfSampleUiBindings& bindings)
{
    AnfSampleUiResult result{};

    if (!bindings.DispatchImmediate ||
        !bindings.MaxFramesInFlight ||
        !bindings.UpscalingEnabled ||
        !bindings.FrameGenerationEnabled ||
        !bindings.AnfSrQualityMode ||
        !bindings.AnimationPaused)
    {
        return result;
    }

    const float baseWindowWidth = 450.0f;
    const float windowWidth = baseWindowWidth * 1.3f;
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const float posX = std::max(20.0f, (displaySize.x - baseWindowWidth - 24.0f) - (windowWidth - baseWindowWidth));
    const float posY = std::max(20.0f, displaySize.y * 0.16f);
    ImGui::SetNextWindowSize(ImVec2(windowWidth, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(posX, posY), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("ANF Sample Controls"))
    {
        if (ImGui::CollapsingHeader("Performance (GPU)", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed))
        {
            const bool dispatchImmediate = *bindings.DispatchImmediate;

            if (!model.ProfilingReady || !model.Profiler)
            {
                ImGui::TextDisabled("GPU timestamps unavailable.");
            }
            else
            {
                const auto& gpuMetrics = model.Profiler->GetMetrics();
                const auto& appScene   = gpuMetrics.Get(AnfGpuProfiler::Region::SceneReal);
                const auto& hudFake    = gpuMetrics.Get(AnfGpuProfiler::Region::HudFake);
                const auto& blitFake   = gpuMetrics.Get(AnfGpuProfiler::Region::BlitFake);
                const auto& srDispatch = gpuMetrics.Get(AnfGpuProfiler::Region::SrDispatch);
                const auto& fgPrep     = gpuMetrics.Get(AnfGpuProfiler::Region::FgPrepareReal);
                const auto& fgDispatch = gpuMetrics.Get(AnfGpuProfiler::Region::FgDispatch);

                const bool srActive = *bindings.UpscalingEnabled && model.SrSupported;
                const bool fgActive = *bindings.FrameGenerationEnabled && model.FgSupported;

                const double appBaseLast = gpuMetrics.AppBaselineLastMs();
                const double appBaseAvg  = gpuMetrics.AppBaselineAvgMs();

                const bool srCostValid = srActive && srDispatch.Valid;
                const double srOverPct = (srCostValid && appBaseLast > 0.0) ? (100.0 * srDispatch.LastMs / appBaseLast) : 0.0;

                const bool fgDispatchValid = fgActive && fgDispatch.Valid;
                const bool fgTotalValid = fgActive && fgPrep.Valid && fgDispatch.Valid;
                const double fgTotalLast = fgPrep.LastMs + (fgDispatchValid ? fgDispatch.LastMs : 0.0);
                const double fgTotalAvg  = fgPrep.AvgMs + (fgDispatchValid ? fgDispatch.AvgMs : 0.0);
                const double fgOverPct   = (fgTotalValid && appBaseLast > 0.0) ? (100.0 * fgTotalLast / appBaseLast) : 0.0;

                bool realFrameValid = appBaseLast > 0.0;
                bool fgFrameValid = fgActive;

                double realFrameLast = appBaseLast;
                double realFrameAvg  = appBaseAvg;
                if (srActive)
                {
                    realFrameValid &= srDispatch.Valid;
                    realFrameLast += srDispatch.LastMs;
                    realFrameAvg += srDispatch.AvgMs;
                }
                if (fgActive)
                {
                    realFrameValid &= fgPrep.Valid;
                    realFrameLast += fgPrep.LastMs;
                    realFrameAvg += fgPrep.AvgMs;
                }

                double fgFrameLast = 0.0;
                double fgFrameAvg  = 0.0;
                if (fgActive)
                {
                    fgFrameValid &= hudFake.Valid && blitFake.Valid && fgDispatch.Valid;
                    fgFrameLast = hudFake.LastMs + blitFake.LastMs + fgDispatch.LastMs;
                    fgFrameAvg  = hudFake.AvgMs + blitFake.AvgMs + fgDispatch.AvgMs;
                }

                if (model.HasCalculatedFps)
                    ImGui::Text("FPS(calc) Real: %.1f | Present: %.1f", model.CalculatedRealFps, model.CalculatedPresentedFps);
                else
                    ImGui::TextDisabled("FPS(calc) Real: - | Present: -");
                ImGui::SameLine();
                ImGui::Text("| Dispatch: %s | MIF: %u", dispatchImmediate ? "Imm" : "Ind", model.ActiveAnfFramesInFlight);
                ImGui::Text("Frame ms (GPU) Real: %.2f | FG: %.2f", realFrameLast, fgFrameLast);

                if (ImGui::BeginTable("PerfTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
                {
                    ImGui::TableSetupColumn("Item");
                    ImGui::TableSetupColumn("Last ms");
                    ImGui::TableSetupColumn("Avg ms");
                    ImGui::TableSetupColumn("Ovhd");
                    ImGui::TableHeadersRow();

                    auto drawRow = [](const char* name, bool valid, double lastMs, double avgMs, bool hasOverhead, double overheadPct, const char* state)
                    {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(name);
                        ImGui::TableSetColumnIndex(1); valid ? ImGui::Text("%.2f", lastMs) : ImGui::TextDisabled("-");
                        ImGui::TableSetColumnIndex(2); valid ? ImGui::Text("%.2f", avgMs)  : ImGui::TextDisabled("-");
                        ImGui::TableSetColumnIndex(3);
                        if (hasOverhead)
                            ImGui::Text("%.1f%%", overheadPct);
                        else
                            ImGui::TextDisabled("%s", state);
                    };

                    drawRow("AppBase", appBaseLast > 0.0, appBaseLast, appBaseAvg, false, 0.0, "-");
                    drawRow("AppScene", appScene.Valid, appScene.LastMs, appScene.AvgMs, false, 0.0, "-");
                    drawRow("SR", srCostValid, srDispatch.LastMs, srDispatch.AvgMs, srCostValid, srOverPct, srActive ? (dispatchImmediate ? "Imm~" : "Ind") : "Off");
                    drawRow("FGPrep", fgActive && fgPrep.Valid, fgPrep.LastMs, fgPrep.AvgMs, false, 0.0, fgActive ? "Wait" : "Off");
                    drawRow("FG", fgDispatchValid, fgDispatch.LastMs, fgDispatch.AvgMs, false, 0.0, fgActive ? (dispatchImmediate ? "Imm~" : "Ind") : "Off");
                    drawRow("FGTot", fgTotalValid, fgTotalLast, fgTotalAvg, fgTotalValid, fgOverPct, fgActive ? (dispatchImmediate ? "Imm~" : "Ind") : "Off");
                    drawRow("RealTot", realFrameValid, realFrameLast, realFrameAvg, false, 0.0, "Mix");
                    drawRow("FGFrame", fgActive && fgFrameValid, fgFrameLast, fgFrameAvg, false, 0.0, fgActive ? "Mix" : "Off");

                    ImGui::EndTable();
                }

                if (dispatchImmediate)
                {
                    ImGui::TextColored(
                        ImVec4(1.0f, 0.55f, 0.35f, 1.0f),
                        "Imm timings are approximate; prefer Indirect.");
                }
            }

            ImGui::TextDisabled("Queue-visible Vulkan GPU time; ANF internal OpenCL may be outside this view.");
        }

        if (ImGui::CollapsingHeader("ANF", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed))
        {
            ImGui::Checkbox("Dispatch Immediate", bindings.DispatchImmediate);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Single ANF instance mode used by both SR and FG.\nImmediate: ANF submits internally.\nIndirect: app records/submits command buffers.");

            ImGui::SliderInt("Max In-Flight", bindings.MaxFramesInFlight, 1, static_cast<int>(model.MaxFramesInFlightLimit));

            if (ImGui::Checkbox("Enable SR", bindings.UpscalingEnabled))
                result.RequestSrReset = true;
            ImGui::SameLine(0.0f, 12.0f);
            ImGui::TextDisabled(model.SrSupported ? "Supported" : "Not Supported");
            const char* qualityModes[] = { "Performance", "Balanced", "Quality" };
            int qualityMode = *bindings.AnfSrQualityMode;
            ImGui::SameLine(0.0f, 14.0f);
            ImGui::TextUnformatted("Quality");
            ImGui::SameLine(0.0f, 6.0f);
            ImGui::BeginDisabled(!model.SrSupported);
            ImGui::SetNextItemWidth(160.0f);
            if (ImGui::Combo("##SRQualityInline", &qualityMode, qualityModes, IM_ARRAYSIZE(qualityModes)))
                *bindings.AnfSrQualityMode = qualityMode;
            ImGui::EndDisabled();

            ImGui::BeginDisabled(!model.FgSupported);
            ImGui::Checkbox("Enable FG", bindings.FrameGenerationEnabled);
            ImGui::EndDisabled();
            ImGui::SameLine(0.0f, 12.0f);
            ImGui::TextDisabled(model.FgSupported ? "Supported" : "Not Supported");

            if (bindings.IgnoreAdbRuntimeCommands)
            {
                ImGui::Checkbox("Ignore ADB Runtime Commands", bindings.IgnoreAdbRuntimeCommands);
                ImGui::SameLine(0.0f, 6.0f);
                ImGui::TextDisabled("?");
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted("When checked, the sample ignores ADB properties and only UI state controls SR/FG.");
                    ImGui::Spacing();
                    if (ImGui::BeginTable("AnfAdbCommandsHelp", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Command");
                        ImGui::TableSetupColumn("Value");
                        ImGui::TableSetupColumn("Effect");
                        ImGui::TableHeadersRow();

                        auto drawCommand = [](const char* command, const char* value, const char* effect)
                        {
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(command);
                            ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(value);
                            ImGui::TableSetColumnIndex(2); ImGui::TextUnformatted(effect);
                        };

                        drawCommand("adb shell setprop debug.anf.dynamic_cmds", "1", "Enable ADB runtime polling");
                        drawCommand("adb shell setprop debug.anf.sr", "1 / 0", "Enable / disable SR");
                        drawCommand("adb shell setprop debug.anf.fg", "1 / 0", "Enable / disable FG");
                        drawCommand("adb shell setprop debug.anf.start_dispatch_immediate", "1 / 0", "Startup-only dispatch mode override");
                        ImGui::EndTable();
                    }
                    ImGui::EndTooltip();
                }
            }

            if (bindings.UseInverseDepth && ImGui::Checkbox("Use Inverse Depth", bindings.UseInverseDepth))
            {
                result.RequestSrReset = true;
                result.RequestFgReset = true;
            }
        }

        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed))
        {
            ImGui::Checkbox("Pause Animation", bindings.AnimationPaused);
        }

        if (ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_Framed))
        {
            if (bindings.DebugForceZeroMv)
                ImGui::Checkbox("Force Zero Motion Vectors", bindings.DebugForceZeroMv);
            if (bindings.DebugForceDisableJitter)
                ImGui::Checkbox("Force Disable Jitter", bindings.DebugForceDisableJitter);
            if (bindings.DebugForceWaitForIdle)
                ImGui::Checkbox("Force Wait For Idle", bindings.DebugForceWaitForIdle);
        }
    }
    ImGui::End();

    return result;
}
