Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, 1, 0, 1.0};
Point(3) = {0.5, 0, 0, 1.0};
Point(4) = {0.5, 1, 0, 1.0};
Point(5) = {0.75, 0, 0, 1.0};
Point(6) = {0.75, 1, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {3, 4};
Line(3) = {2, 4};
Line(4) = {1, 3};
Line(5) = {5, 6};
Line(6) = {3, 5};
Line(7) = {4, 6};
Line Loop(8) = {1, 3, -2, -4};
Plane Surface(9) = {8};
Line Loop(10) = {7, -5, -6, 2};
Plane Surface(11) = {10};
Physical Line(12) = {1, 3, 7, 5, 6, 4};
Physical Surface(1) = {9};
Physical Surface(2) = {11};
