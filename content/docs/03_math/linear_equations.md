---
weight: 302
title: "Linear Equations"
---

# Linear Equations

{{< katex display >}}
x = A^{-1} \cdot b
{{< /katex >}}

## Example A

{{< katex display >}}
\tag{a.1} x + 2y = 5
{{< /katex >}}

{{< katex display >}}
\tag{a.2} 3x + 4y = 6
{{< /katex >}}

{{< katex display >}}
\tag{a.3} A =
\begin{pmatrix}
1 & 2 \\
4 & 3
\end{pmatrix}

\quad

b =
\begin{pmatrix}
5 \\
6
\end{pmatrix}
{{< /katex >}}

{{< katex display >}}
\tag{a.4} x =
\begin{pmatrix}
1 & 2 \\
4 & 3
\end{pmatrix}^{-1}

\cdot

\begin{pmatrix}
5 \\
6
\end{pmatrix}
{{< /katex >}}

### Python Solution

```python
import numpy as np
import numpy.linalg as la

A = np.array([[1, 2], [3, 4]])
b = np.array([[5], [6]])

Ainv = la.inv(A)
x = Ainv.dot(b)
```
