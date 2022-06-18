# Navier_Stokes_sphere2D
Physics Informed Neural Network (PINN) for the 2D Navier-Stokes equation

This module implements the Physics Informed Neural Network (PINN) model for the 2D Navier-Stokes equation. The NS equations are given by (du/dx + dv/dy) = 0, u du/dx + v du/dy + dp/dx - (d^2u/dx^x + d^2u/dy^2) / Re = 0, u dv/dx + v dv/dy + dp/dy - (d^2v/dx^2 + d^2v/dy^2) / Re = 0. It represents the fluid flow over a cylinder inside a wind tunnel depending on the Reynolds number. The PINN model predicts u(x, y) for the input (x, y).

It is based on hysics informed neural network (PINN) for the 1D Wave equation on https://github.com/okada39/pinn_wave
