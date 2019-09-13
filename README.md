# w2vxx
w2vxx — набор утилит для построения векторных представлений лексики какого-либо естественного языка. Фактически — это [word2vec](https://ru.wikipedia.org/wiki/Word2vec), реализованный на C++.

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
В состав w2vxx входит четыре утилиты: build_dict, cbow, skip-gram и distance. В отличие от word2vec построение словаря здесь выделено в отдельную подзадачу (build_dict), а различные модели обучения — cbow и skip-gram — реализованы в одноимённых утилитах.

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
