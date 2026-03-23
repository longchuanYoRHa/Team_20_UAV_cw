# AERO60492 Coursework 3 – UAS Feedback Control

## Overview 

This repository contains the implementation of a feedback control algorithm for position stabilisation of an Unmanned Aerial System (UAS), developed as part of the AERO60492 – Autonomous Mobile Robots coursework.

The project focuses on designing a controller that processes sensor feedback (position, velocity, attitude) and generates actuation commands to drive the UAV to a desired target position.

## Project timeline

```mermaid

gantt
    title AERO60492 Coursework Timeline
    dateFormat  YYYY-MM-DD

    section Setup
    Coursework Released           :done,    a1, 2025-03-10, 3d
    Simulator Setup & Familiarisation :done, a2, after a1, 4d

    section Implementation
    Controller Design             :active,  b1, 2025-03-17, 5d
    Implementation in Simulator   :         b2, after b1, 7d

    section Testing
    Initial Testing & Data Collection :     c1, 2025-03-31, 5d
    Controller Tuning & Refinement    :     c2, after c1, 5d

    section Finalisation
    Final Testing                 :         d1, 2025-04-14, 3d
    Video & Report Preparation    :         d2, after d1, 5d

    section Submission
    Submission Deadline           :milestone, m1, 2025-05-15, 0d
```