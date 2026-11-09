# Evaluación de ataques A(c, beta)

Extensión sobre la actualización del 6 de noviembre de 2026. Conserva los módulos
actuales de datos, vecinos, generalización y métricas. Añade `kscale/attacks.py` y
`kscale_attacks.ipynb`; no requiere repetir la anonimización.

## Ejecución

Descomprime el paquete en una carpeta nueva, entra en ella y ejecuta:

```bash
python -m pip install -e .
```

Abre `kscale_attacks.ipynb`, configura `output_path` con el directorio de tu ejecución
existente y ejecuta las celdas. También se incluyen las celdas de ataque al final de
`kscale_pipeline.ipynb` para futuras ejecuciones completas.

```python
from kscale import simulate_attacks
result = simulate_attacks(
    Sigma='run_metrics_01/Sigma',
    Sigma_tilde='run_metrics_01/k_5/Sigma_tilde',
    output_path='run_metrics_01/k_5/attacks_01',
    c_values=[1, 2, 3, 5], beta_values=[1, 2, 4, 8, 'all'],
    trials=100, seed=42,
)
```

Empieza con 10--100 repeticiones; aumenta a 1000 después de medir el tiempo.
Una ventana de una hora en YJMob suele no permitir beta=8. El notebook propone
beta=[1,2,'all'] y comprueba el tamaño de la cohorte. El módulo rechaza cohortes
insuficientes: nunca reduce beta silenciosamente. Todos los beta, incluido 'all',
usan usuarios con al menos max(beta numérico) muestras. Cambiar ese máximo cambia
la población evaluada. 'all' revela todos los puntos de cada víctima, no el mismo
número de puntos para todos. Las muestras son filas, aunque haya coordenadas repetidas.

## Modelo y reproducibilidad

- `c`: víctimas distintas conocidas. `beta`: puntos conocidos por víctima.
- `random`: víctimas uniformes sin reemplazo dentro de la cohorte.
- `nearby`: ancla aleatoria y vecinos por centroide (x,y,t) con escalas explícitas
  `spatial_scale_m=1000`, `time_scale_s=3600`. Es una aproximación a proximidad;
  no garantiza encuentros simultáneos ni relaciones sociales.
- Puntos aleatorios sin reemplazo o prefijos de un tramo temporal consecutivo.
  El tramo se elige con longitud max(beta numérico); los tramos más cortos son
  prefijos de ese tramo, no ventanas uniformes independientes.
- Se reutilizan prefijos de víctimas y puntos: conocer más no aumenta candidatos.
- La misma Sigma, semilla y configuración de muestreo reutiliza selecciones entre
  diferentes k/K. Conservar Sigma es esencial para mantener IDs y población.
- Se comprueba el dataset_id, la cobertura >= k y la inclusión de todos los hosts
  en los candidatos observables de cada consulta. La comprobación de contención
  es sobre los puntos consultados. Para verificar todos los puntos de todos los
  miembros utiliza antes `calculate_metrics(..., verify_containment=True)`.

## Resultados

- `attacks.csv`: una fila por ataque; víctimas, índices de puntos en orden temporal,
  c, beta, semilla, gamma estructural, J observado y número total de puntos conocidos.
- `attack_summary.csv`: medias, mediana y tasas de candidato único, cero candidatos,
  colapso 0<J<k, colapso condicionado a J>0, solapamiento y colapso estructurales.
  Incluye numeradores, denominadores e intervalos Wilson del 95%.
- `structure.csv`: cobertura real y tamaño de grupo por usuario/registro. La reparación
  de filas vacías de nu-k-pick puede aumentar la cobertura por encima de k.
- `metadata.json`: parámetros, población elegible, identidad del dataset y rutas.

Los intervalos describen error Monte Carlo condicionado a la publicación fija.
Los escenarios están emparejados; no trates todas las filas de todos los escenarios
como observaciones independientes para comparar configuraciones.

## Alcance de las conclusiones

Se mide compatibilidad geométrica: cada punto conocido debe estar contenido en
alguna caja del registro, incluyendo fronteras. Se buscan TODOS los registros,
no solo H(i). A.npz es información del evaluador, no información del atacante.
El atacante no usa N_i_segment, L, costes, etiquetas de propietario, correspondencias
uno-a-uno de muestras, restricciones dinámicas ni inferencias del algoritmo de selección.
Estas limitaciones importan al interpretar resultados de privacidad.

Gamma es el número de registros cuyos miembros contienen a todas las víctimas;
J incluye cobertura accidental. Siempre J >= gamma. Con beta creciente gamma
permanece fijo y J no aumenta. J=1 significa un único registro conjunto compatible,
NO identificación de una persona, NO probabilidad posterior 1/J y NO reconstrucción.
J=0 se reporta, no se interpreta como garantía de ausencia de otros ataques.

## Conocimiento complementario: ejemplo exacto y limitado

`single_point_compatible_values` enumera mundos de una rejilla finita cuando cada
usuario aporta un punto y se publica solo la caja min/max de ese grupo. Admite
varios desconocidos etiquetados y devuelve posiciones compatibles de C, el primero.
No calcula probabilidades sin una distribución previa y no invierte gen para
trayectorias completas. La enumeración está limitada por `max_worlds`.

## Coste

El índice carga todas las cajas de la publicación en RAM y examina las cajas por
bloques para cada punto no cacheado. No construye una matriz U x U densa. La caché
LRU está limitada por entradas (`cache_size`), no por bytes; sus conjuntos pueden
ser grandes en publicaciones muy solapadas. Las trayectorias crudas tienen caché de
256 usuarios. Los ataques y sus índices de puntos se mantienen en memoria hasta
terminar. Usa muestras de usuarios para una primera evaluación. No se ha validado
rendimiento sobre los 100000 usuarios completos.

Las salidas son etapas atómicas y no se sobrescriben. Cambia el directorio de salida
para otra ejecución. Este paquete incluye el pipeline anterior completo, no datos.

## Pruebas

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

Las pruebas nuevas incluyen pipeline real en datos sintéticos, búsqueda por fuerza
bruta, reproducibilidad, monotonía, comprobaciones de compatibilidad, fronteras,
mapas de calor y ejemplos de reconstrucción/ambigüedad del punto único.
