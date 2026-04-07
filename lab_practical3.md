# Practical 3: pick-and-place application

::: highlight
##### Overview 

In this practical, we apply what we have learned in inverse kinematics to solve a pick-and-place task, and we will also see how to use global optimization in a real-world problem. 

You will tune a pick and place policy, evaluate how well it transfers to the real robot, and reflect on how this problem could be solved with global optimization methods.

:::

:::: collapse Simulation Framework 

In order to test our algorithm, we create a simple simulation environment. 
This setup is intentionally simplified: it is not a high-fidelity grasp simulation. 
A small virtual block is introduced, and a simple grasp logic (no real physics contact dynamics).
The goal is to give you a fast and controllable environment where you can programmatically test and compare different motion profiles before trying them on hardware.

::: exercise
**Exercise 1: tune the simulation policy**

Goal: make the pick-and-place sequence reliable in simulation.

Open `practical3.py` and tune the selected parameter dictionary inside `_default_task_tuning()`. Do not change controller logic for this exercise. After you change the parameters, you can run the script using the following button:

#runsofa-button("assets/labs/Practical3/practical3.py")

Tune the following parameters:

1. Gripper setpoints (mm): `gripper_opening_open`, `gripper_opening_closed`
2. Cartesian targets (mm): `pick_position`, `place_position`
3. Vertical offsets (mm): `hover_lift_height`, `pick_height_offset`, `place_height_offset`
4. Optional setup parameter: `object_position`

A good order is:

1. First center pickup with `pick_position`.
2. Then tune `gripper_opening_closed` so pickup is repeatable.
3. Then adjust `hover_lift_height` to avoid collisions during transfer.
4. Finally tune `place_position` for reliable release at the goal.

In your report, include the following: 

a) working parameters 

b) screenshots of the simulation working.

:::

::::

:::: collapse Real-world Evaluation

Once you have found a good set of parameters, you can move over to the real-world evaluation.  

::: exercise
**Exercise 2: transfer to the real robot**

Run your tuned policy on the real robot.

In your report, include a discussion of the following:

1. Which parameters transferred well from simulation to hardware.
2. Which parameters required retuning.
3. One likely source of sim-to-real mismatch you observed.
:::

::::

:::: collapse Optimization

Finally we ask you reflect about how you could improve the above process using a global optimization framework. 

The simulator includes a scoring function, where the score is based on three metrics:

1. Did the robot lift the block?
2. Did the robot place it near the goal position?
3. How quickly was the task completed?

In the following, you can assume that the same metrics can be evaluated on the real robot. 

::: exercise
**Exercise 3: optimization reflection**

In your report, Write a short reflection (5-8 sentences):

1. How you would formulate this tuning task as a global optimization problem.
2. Which variables you would optimize and what objective(s) you would use.
3. Which method is a better fit here, Bayesian Optimization or CMA-ES, and why.

In your argument, consider practical constraints such as noisy evaluations, limited experiment budget, parameter dimensionality, and cost landscape.
:::

::: exercise
**Exercise 4 (bonus):**

If you are feeling adventurous, you can implement a little global optimization procedure to actually solve this problem to global optimality. If you do this task, you can gain up to 15 points that will count towards Homework 3. 

To gain full points, we ask you to discuss the following in your report: 
1. Include a screenshot and a link to the codebase of your working code. (5pt) 
2. Provide a convergence plot, and (3pt) 
3. Show screenshots of the working pick and place task after finding your (2pt)
4. Discuss the performance and its limitations (5pt) 
:::

::::
