# Contributing to the Adreno™ GPU Vulkan code sample framework

Use issues to report a reproducible problem and pull requests to propose a focused change. Read the [code of conduct](CODE-OF-CONDUCT.md) and [license](LICENSE.txt).

## Report a problem

Include the sample, repository revision, build command, platform, device and driver version, expected result, and observed result. Add relevant logs without credentials, private paths, or proprietary content.

## Prepare a change

1. Create a branch based on `main`.
2. Keep the change focused on the reported problem or feature.
3. Build the affected samples and test on an Android device when the change affects Android. For shared framework changes, run an existing test application too.
4. Update affected sample and test guides when behavior, requirements, controls, or build steps change.
5. Include commands and results in the pull request. State which platforms or devices were not tested.

## Submit a pull request

Commit with a [Developer Certificate of Origin](https://developercertificate.org/) sign-off using `git commit --signoff`. Open a pull request against `main` and describe the problem, resulting behavior, validation, and compatibility impact.

Preserve copyright notices and component-specific licenses. Do not include generated output, local configuration, private SDKs, or assets without redistribution rights.

## Documentation

Use sentence-case headings, short paragraphs, and direct instructions. Format API names, paths, commands, and configuration keys with backticks. Explain required settings and the failure caused by an incorrect value. Link shared setup instructions instead of repeating them in every sample.
