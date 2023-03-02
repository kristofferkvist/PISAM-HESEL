#include "intercomm_monitor.hxx"
#include <bout/solver.hxx>

IntercommMonitor::IntercommMonitor(Options* opt) {
  // Get options
  Options* global_options = Options::getRoot();
  Options* options = opt == nullptr ? global_options->getSection("kinetic_neutrals")
                                    : opt;
  int kinetic_neutral_dynamics;
  OPTION(options, kinetic_neutral_dynamics, 0);
  if (kinetic_neutral_dynamics == 1) {
    enabled = true;
    // Calculate output frequency
    BoutReal output_timestep;
    global_options->get("timestep", output_timestep, 1.);
    int frequency_multiplier;
    OPTION(options, frequency_multiplier, 10); // multiple of the output frequency to call fast_output at
    setTimestep(output_timestep / double(frequency_multiplier));
  }
}

int IntercommMonitor::monitor_method(BoutReal simtime) {
  // Set time
  current_time = simtime;

  //print time
  std::cout << "######################################SIMTIME FROM INTERCOMM MONITOR" << current_time << std::endl;

  return 0;
}
