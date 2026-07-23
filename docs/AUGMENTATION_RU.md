# Boundary augmentation

Аугментация включается только для train:

- erosion удаляет ограниченную локальную полосу shell boundary;
- dilation рассматривает fracture pixels и depth ring около границы;
- mixed последовательно выполняет оба действия.

Для кандидата `p_C`:

```text
p_O_raw = inverse(T_C_from_O_GT) · p_C
q_O = closest_point_on_template(p_O_raw)
accept ⇔ ||p_O_raw - q_O|| ≤ d_template
         ∧ |depth(candidate)-local_shell_depth| ≤ d_depth
```

После принятия модель получает `p_C`, а train target — `q_O`. GT pose нужен
только внутри target construction. Points, targets, UV и labels изменяются
синхронно; затем выполняется max-point sampling. Val/test augmentation
принудительно запрещена.
