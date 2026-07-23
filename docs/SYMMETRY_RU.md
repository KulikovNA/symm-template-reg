# Симметрия

Sidecar version 1 задаёт object frame `O`, ось с origin/direction и
неперекрывающиеся axial regions с группами `Cn` или `SO2`.
Активные regions определяются каноническими точками фрагмента; effective group
равна пересечению их групп (`Cn ∩ Cm = C_gcd(n,m)`, `SO2` не ограничивает
конечную группу).

`tools/visualize_template_symmetry.py` раскрашивает face regions, сохраняет
legend, вычисляет active regions/effective group и gallery. В цветном PLY
треугольники разъединены на границе, поэтому shared vertices не смешивают цвета.
