EMG-CONTROLLED ROBOTIC HAND
==========================

This repository contains almost all the working code and data I used to develop my EMG-controlled robotic hand for my final-year engineering project.

The idea is simple: use muscle signals from your forearm to control a 3D-printed robotic hand.

The system uses four EMG sensors, an Arduino Nano ESP32, a machine-learning classifier and five servomotors to move the fingers.

The final system recognises four gestures:
- Hand close
- Index finger up
- Pinch
- Rock'n'roll

Relaxing the forearm opens the hand.

The final prototype achieved 73 successful gestures out of 80 attempts (91.25%) during one live testing session.

1. WHAT YOU NEED
----------------

Electronics:
- Arduino Nano ESP32 (1)
- MyoWare 2.0 EMG sensors (4)
- EMG electrodes (3 per sensor)
- DS9G-270-FB 270-degree servomotors (5)
- 18650 Li-ion batteries (2)
- 2S battery management system / BMS (1)
- 5 V DC-DC buck converter (1)
- Connecting wires and connectors (as needed)

The servos are powered separately from the microcontroller using the 2S battery supply. The microcontroller uses a regulated 5 V supply.

IMPORTANT: Check the voltage ratings, battery protection, wiring and common ground connections before powering anything. Do not power all five servos directly from the Arduino.

Mechanical parts:
I used the InMoov i2 hand, developed by Gael Langevin.
Official website: https://inmoov.fr/inmoov-hand/

The main parts you need include:
- Five printed fingers and fingertips
- Palm and hand covers
- Wrist connection parts
- Five servo pulleys and servo adapters
- Tendon cables
- Finger joints, pins and fasteners
- PLA filament for printing

The exact printed parts depend on the hand version and servo arrangement you choose. Download the matching STL files and follow the instructions on the InMoov website. Credit for the original mechanical design belongs to Gael Langevin; check the licence of the files you use before redistributing them.

2. SOFTWARE
-----------

You will need:
- Python on your computer: https://www.python.org/
- Thonny IDE: https://thonny.org/
- MicroPython installed on the Nano ESP32
- Python libraries required by the training scripts, including NumPy and scikit-learn

3. HOW TO REPRODUCE THE CONTROL SYSTEM
-------------------------------------

Step 1: Assemble the hand
Print and assemble the InMoov parts, install the five servomotors and connect the tendons. Test finger movements before trying EMG control. Make sure each servo moves in the correct direction and its travel limits do not damage the fingers or tendons.

Step 2: Connect the sensors
Connect the four MyoWare sensors to the Arduino Nano ESP32. I used the RAW EMG outputs and sampled four channels at approximately 1 kHz per channel. I placed the sensors over forearm muscle regions associated with finger flexion, wrist flexion, thumb movement and finger extension. You may need to experiment with positioning to get good signals.

Step 3: Record your own EMG data
Open Thonny, connect to your Nano ESP32 and run:

    recordsesion.py

This is the recording script I used to collect data from my arm. My recording protocol was:
1. Rest for 3 seconds.
2. Perform the gesture for 3 seconds.
3. Rest for another 3 seconds.
4. Repeat for the other gestures and recording cycles.

I randomised the gesture order during recording. You can follow my protocol or change it if you have a better approach. Just keep the labels and recording format consistent.

I strongly recommend recording your own data. EMG signals change between people, electrode positions and sessions.

Step 4: Train the LDA classifier
Transfer the recorded data to your computer. Run the LDA training script (labelled "fit and save lda" in my workflow) using Python on your computer. Check the actual filename in the repository.

My final classifier used four time-domain features for each EMG channel:
- Mean Absolute Value (MAV)
- Waveform Length (WL)
- Zero Crossings (ZC)
- Slope Sign Changes (SSC)

Four channels give 16 features per window. My final settings used a 150-sample window and a 38-sample step (about 75% nominal overlap). I used separate recording cycles for training and testing rather than randomly mixing overlapping windows.

Step 5: Export the model parameters
After training the LDA model, export its parameters. Copy the exported weights and any other required model parameters into the MicroPython implementation on the Nano ESP32.

IMPORTANT: Feature order, preprocessing, thresholds and class order must match between training and deployment. Otherwise, the model may run but return incorrect predictions.

Step 6: Run the controller
Upload the required MicroPython files to the Nano ESP32 using Thonny. Run the main EMG controller script.

Pipeline:
EMG sensors -> sampling -> feature extraction -> LDA classification -> decision filtering -> servo movement

I used a rest threshold, switching margins and debounce logic to reduce unwanted movements. You may need to recalibrate these settings for your arm.

4. A FEW THINGS WORTH KNOWING
----------------------------

This is a research prototype, not a finished commercial prosthetic hand. The final model was trained and tested using my own forearm signals. I can't guarantee it will work immediately for another person.

The system uses sequential sampling and processing: sampling pauses while the microcontroller calculates features and makes predictions. That affects response time.

Continuous sampling, better electrode mounting and testing with multiple users would be good improvements.

5. FINAL ADVICE
---------------

Don't be surprised if your first classifier gives terrible results. Mine certainly did.

Check your signals, check the labels and make sure you're not accidentally training on rest periods.

Also, don't trust offline accuracy too much. A classifier can perform well on recorded data and still behave badly when controlling the actual hand.

Feel free to experiment with the code, change the recording protocol, try other classifiers or improve the mechanical design. That's partly why I decided to share the project. If someone can reproduce it, learn something from it or make a better version, even better.

I am putting this stuff not even knowing if I am gonna pass (So this might be trash or good)


If I did make this stuff,evryone can ;)


Good luck, and have fun building it :) !!!