# w2vxx
w2vxx — набор утилит для построения векторных представлений слов какого-либо естественного языка. Фактически это [word2vec](https://ru.wikipedia.org/wiki/Word2vec), реализованный на C++.

w2vxx создан с целью повысить модульность и упростить код оригинального word2vec. Надеюсь, что w2vxx позволит сэкономить усилия экспериментаторов и разработчиков, желающих усовершенствовать алгоритмы, лежащие в основе word2vec.

## Быстрый старт
В репозитории размещены демонстрационные скрипты для Linux и Windows, обеспечивающие:
1. сборку утилит из исходных кодов, 
2. загрузку (англоязычного) обучающего множества,
3. построение словаря и векторной модели,
4. запуск утилиты, которая для заданного слова отыскивает в модели близкие по значению слова.

Для сборки утилит требуется компилятор с поддержкой [C++17](https://ru.wikipedia.org/wiki/C%2B%2B17). Сборка протестирована под Linux с компилятором gcc v7.4.0 и под Windows с компилятором от Visual Studio 2017 v15.9.14 (cl.exe версии 19.16).

<table>
  <tr>
    <th width="50%">Запуск под Linux</th>
    <th>Запуск под Windows</th>
  </tr>
  <tr>
    <td valign="top">Запустите консоль. Перейдите в директорию, в которой хотите развернуть программное обеспечение.</td>
    <td>Запустите «Командную строку разработчика для VS 2017» (это обеспечит настройку окружения для сборки утилит). Перейдите в папку, в которой хотите развернуть программное обеспечение.</td>
  </tr>
  <tr>
    <td colspan="2" align="center">git clone https://github.com/w2vxx/w2vxx.git<br/>cd w2vxx</td>
  </tr>
  <tr>
    <td align="center">./demo-linux.sh</td>
    <td align="center">demo-windows.cmd</td>
  </tr>
  <tr>
    <td colspan="2">При успешных сборке и обучении работа скрипта завершится вызовом утилиты, отыскивающей слова с близким значением. Чтобы удостовериться в работоспособности утилит, попробуйте ввести распространённые английские слова — phone, car, king. Отмечу, что демонстрационный скрипт порождает неоптимальную векторную модель, чтобы сократить время обучения.</td>
  </tr>
</table>

## Утилиты и их параметры
В состав w2vxx входит четыре утилиты: build_dict, cbow, skip-gram и distance. В отличие от word2vec, построение словаря здесь выделено в отдельную подзадачу (build_dict), а различные модели обучения — cbow и skip-gram — реализованы в одноимённых утилитах.

### build_dict
Решает задачу построения словаря по обучающему множеству. Параметры утилиты:

<table>
  <tr>
    <td>-train</td><td>имя файла, содержащего обучающее множество;</td>
  </tr>
  <tr>
    <td>-min-count</td><td>частотный порог. Слова, частота которых (в обучающем множестве) ниже порога, не попадают в словарь;</td>
  </tr>
  <tr>
    <td>-save-vocab</td><td>имя файла, куда будет сохранён словарь.</td>
  </tr>
</table>

### cbow
Осуществляет построение векторных представлений слов языка в соответствии с моделью обучения Continuous Bag-of-Words (cbow). Параметры утилиты:

<table>
  <tr>
    <td>-train</td><td>имя файла, содержащего обучающее множество;</td>
  </tr>
  <tr>
    <td>-words-vocab</td><td>имя файла, содержащего словарь, построенный утилитой build_dict;</td>
  </tr>
  <tr>
    <td>-output</td><td>имя файла, куда будут сохранены векторные представления слов. Файл имеет бинарный формат, полностью совместимый с word2vec;</td>
  </tr>
  <tr>
    <td>-size</td><td>размерность результирующих векторов для представления слов (размерность эмбеддинга);</td>
  </tr>
  <tr>
    <td>-window</td><td>размер окна, задающего контекст слова;</td>
  </tr>
  <tr>
    <td>-optimization</td><td>метод оптимизации вычислений. Значение hs соответствует hierarchical softmax, значение ns соответствует negative sampling. В отличие от оригинального word2vec, можно использовать только один из методов;</td>
  </tr>
  <tr>
    <td>-negative</td><td>для метода negative sampling количество отрицательных примеров, противопоставляемых каждому положительному примеру (количество слов, выбираемых из noise distribution);</td>
  </tr>
  <tr>
    <td>-iter</td><td>количество эпох обучения;</td>
  </tr>
  <tr>
    <td>-sample</td><td>коэффициент прореживания. Обеспечивает снижение в обучающем множестве доли частотных слов. По умолчанию 1e-3;</td>
  </tr>
  <tr>
    <td>-alpha</td><td>начальное значение скорости обучения;</td>
  </tr>
  <tr>
    <td>-threads</td><td>количество потоков управления, параллельно выполняющих обучение модели.</td>
  </tr>
</table>

### skip-gram
Осуществляет построение векторных представлений слов языка в соответствии с моделью обучения Skip-gram. Набор параметров утилиты совпадает с параметрами для cbow.

### distance
Интерактивная утилита для поиска слов, характеризующихся общностью значений. При построении моделей с малым контекстным окном в первую очередь проявляется категориальное сходство (синонимы, антонимы и согипонимы). Если при обучении модели окно было большим, то тематическая и ассоциативная общность также становится значимой.

Для каждого введённого пользователем слова утилита находит в векторной модели близкие по значению слова, а также показывает количественную меру близости ([косинусная мера](https://en.wikipedia.org/wiki/Cosine_similarity)). Параметры утилиты (задаются порядком следования):

<table>
  <tr>
    <td>имя файла с векторными представлениями слов, построенными утилитами cbow или skip-gram;</td>
  </tr>
  <tr>
    <td>количество выводимых на экран слов с близкими значениями.</td>
  </tr>
</table>
