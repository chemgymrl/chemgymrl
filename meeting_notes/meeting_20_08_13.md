# Meeting Notes

## Date: August 13, 2020

### Attendees

- Mitchell Shahen (Secretary)
- Isaac Tamblyn
- Sriram Ganapathi
- Nouha Chatti
- Chris Beeler
- Mark Crowley

- Meeting concluded at 9:40
- Isaac and Mark continued the meeting

### Modifications to Lab Manager

- Updating the min/max of the vessel contents should eventually be learned
- Use the analogy of the scientist labelling beakers with their contents (could do a helpful cartoon)

### Modifications to Extract Bench

- The lab manager is to provide additional information to the extraction bench when assigning tasks/vessels. This information includes everything the lab manager knows about the vessel, including the min/max estimation of each material.

### Modifications to React Bench

- Include the ODE Solver Method and proper documentation
- Replace the ODE Solver Method with the SciPy ODE Solver which allows for larger time-steps, while keeping numerical stability
- It is trivial to add a new reaction once people see how the reactions have already been implemented

### Adding a Characterization Bench

- Add an additional bench to analyze the contents of a vessel to update the lab manager's perception of a vessel by providing a spectrum.
- Additionally, the lab manager should be penalized for not knowing exactly what's in the vessel being passed to the benches.
