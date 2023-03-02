#include <bout/monitor.hxx>
#include <bout/physicsmodel.hxx>
class Solver;

// Subclass of BOUT++ Monitor class so we can pass to Solver::
class IntercommMonitor : public Monitor {
  public:
    IntercommMonitor(Options* opt = nullptr);
    virtual ~IntercommMonitor() = default;
    int monitor_method(BoutReal simtime);

    bool enabled=false;

    /// provide Monitor::call method which is called by the Solver
    int call(Solver* UNUSED(solver), BoutReal time, int UNUSED(iter), int UNUSED(nout)) {
      return monitor_method(time);
    }
  private:
    BoutReal current_time;
};
