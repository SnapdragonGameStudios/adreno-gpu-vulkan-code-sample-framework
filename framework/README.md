# Framework

The framework provides shared application setup and teardown, platform events, rendering helpers, and resource management.

A sample links the relevant framework library and implements `Application_ConstructApplication()`. It returns an application derived from `FrameworkApplicationBase` or an existing application helper.

```cpp
class Application : public FrameworkApplicationBase
{
public:
    Application();
    ~Application() override;
};

FrameworkApplicationBase* Application_ConstructApplication()
{
    return new Application();
}
```

Use an existing [test application](../tests/README.md) to see initialization, rendering, and teardown in context. Keep sample-specific rendering in the sample and reusable behavior in the relevant framework component.

When adding a target, supply its `CMakeLists.txt` and register its selection in `Config.txt`. Copying a folder alone does not enable it.

See the [root build instructions](../README.md#configuring) for targets and dependency setup.
