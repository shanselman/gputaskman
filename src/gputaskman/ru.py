"""Human-reviewed Russian UI catalogue. Technical identifiers stay in English."""

MESSAGES = {}

MESSAGES.update({
    "Show readings on taskbar": "Показатели на панели задач",
    "Taskbar panel unavailable: {0}": "Панель показателей недоступна: {0}",
    "No room for readings on the primary horizontal taskbar": "Недостаточно места на основной горизонтальной панели задач",
    "Could not save taskbar position: {0}": "Не удалось сохранить положение панели: {0}",
    "hide the always-visible taskbar readings (Windows)": "скрыть постоянно видимые показатели на панели задач (Windows)",
    " … (+{0} GPU)": " … (ещё GPU: {0})",
})

HELP_RU = """\
# Что можно узнать о памяти GPU?

**m** или **Кратко/Подробно** переключает подробную диагностику и краткий мониторинг.
В кратком режиме скрыты аппаратные идентификаторы и второстепенная телеметрия.
Текущие показатели, графики, процессы, инспектор и предупреждения об источниках
и доступности данных остаются видимыми. Режим применяется ко всем GPU; после
перезапуска снова включается подробная диагностика.

| Что показывается | Источник | Где доступно |
|---|---|---|
| Телеметрия устройства | NVML, AMD ADL, необязательная AMD SMI, активность Intel WDDM в Windows; sysfs amdgpu или Intel i915 в Linux | зависит от GPU и драйвера |
| VRAM, общая ОЗУ и загрузка движков процесса | счётчики Windows WDDM (dxgkrnl); DRM fdinfo amdgpu/Intel i915 в Linux | процессы, видимые WDDM или DRM |
| Среда выполнения (CUDA, HIP, Vulkan, D3D12, ggml, ONNX, PyTorch…) | библиотеки, отображённые в память процесса | все процессы с доступными сведениями |
| Файл модели | отображённые файлы и параметры запуска | llama.cpp, stable-diffusion.cpp, LM Studio, Ollama |
| Имя и тип модели (LLM, принятие решений, эмбеддинги, речь) | заголовок GGUF, хранилища Ollama/LM Studio/Hugging Face, локальные API | семейство GGUF |
| Разбивка на веса, KV-кэш и вычисления | расчёт по GGUF, журналы llama.cpp/Ollama, локальные API | семейство GGUF/llama.cpp |
| Имена отдельных текстур и буферов | отладчик PIX или Nsight, подключённый к приложению | игры и 3D-приложения |

**Основание** показывает, как обнаружена модель: `api` — среда сообщает о загрузке;
`mapped` — файл сейчас отображён в память процесса; `cmdline` — распознан параметр
запуска модели; `path-arg` — при запуске передан путь к файлу, похожему на модель,
но загрузка не доказана; `installed` — кандидат в папке приложения.
Два источника из `api`, `mapped`, `cmdline`, включая хотя бы `api` или `mapped`,
дают отметку «подтверждено». Одного `path-arg` недостаточно для определения среды,
опроса API или оценки весов/KV-кэша. Настройки приложения означают **выбор** модели,
а записи журнала — **историю**, а не текущее размещение в памяти GPU.

**Только локальные задачи:** инференс, модели принятия решений, зрение, речь,
обучение, графика и видео. Облачные модели здесь не отслеживаются.
Библиотеки определяют среду выполнения, но не модель и не задачу.
`decision?` / «предположительно модель принятия решений (по имени)» — подсказка
по имени файла или папки, если точный тип GGUF неизвестен. Она не подтверждает
тип, размещение в GPU или выполнение. Установленные кандидаты остаются
неподтверждёнными. Загрузка движков измеряется по процессу, а не по модели.

**Linux:** на NVIDIA VRAM процесса и загрузка SM/кодирования/декодирования
получены из NVML. Счётчиков общей и выделенной памяти нет; соответствующие
столбцы остаются пустыми. **AMD amdgpu** использует sysfs для устройства и
`/proc/<pid>/fdinfo` для процессов: VRAM, GTT (память GPU в системной ОЗУ,
столбец «Общая»), время графических, вычислительных и видеодвижков. Для первого
замера клиента загрузка — `n/a`. Для ROCm/HIP она также `n/a`: ядро не измеряет
вычислительные очереди `/dev/kfd` по процессам.

VRAM и общая память относятся к процессу только при доказанно приватных
DRM-клиентах: один владелец, узел render и отсутствие общих буферов. Иначе —
`n/a`, байты остаются нераспределёнными. Инспектор показывает доступную процессу
резидентную память, включая общие буферы; такие значения **не суммируются**.

**Intel i915 в Linux** использует тот же путь: sysfs для фактической частоты GT
и причин ограничений, fdinfo для движков и локальной памяти дискретных Arc.
На встроенном GPU системная память доказанно приватного клиента показывается
в VRAM; «Общая» — `N/A`, поскольку системная память и есть его память GPU.
Для общих клиентов — `n/a`, а резидентные байты видны в инспекторе.
Карты `xe` обнаруживаются, но fdinfo процессов пока не декодируется (`n/a`).
Общего счётчика памяти Intel нет: занятая VRAM — сумма видимых процессов
(локальная память дискретного GPU или системная встроенного). Общие буферы
могут учитываться дважды; память ядра и недоступных процессов не включена.
Общий объём — оценка по размеру PCI-окна локальной памяти, неизвестная для
встроенного GPU. Свободная память не показывается. Заголовок сообщает число
недоступных процессов; `sudo` позволяет включить процессы других пользователей.

**WSL:** Windows относит всю память GPU в Linux к ВМ (`vmwp.exe`). Без счётчиков
хоста VRAM и загрузка доступны только для устройства. Строки процессов с
открытым `/dev/dxg` показывают `n/a`, поскольку их доли неизвестны.
При включённом interop счётчики Windows читаются через `powershell.exe`:
«ВМ WSL» (`vmwp.exe`) — точный итог памяти GPU Linux во всех дистрибутивах;
процессы Windows имеют PID `W:<pid>`. `--no-host` отключает этот опрос.
Контексты GPU ВМ сопоставляются с процессами Linux от новых к старым:
`*` — подходит одно объяснение либо все объяснения дают **в точности одинаковый**
объём; `?` — владелец выбран эвристически. Размер контекста точный,
принадлежность процессу определена косвенно. Кэш внешних размеров только
упорядочивает предположения и не доказывает принадлежность между запусками.
Строки Linux входят в итог ВМ и повторно к нему не прибавляются.

**Почему nvidia-smi показывает N/A?** GeForce в Windows работает в WDDM:
видеопамятью управляет Windows, поэтому NVML не может распределить её
по процессам. Счётчики Windows могут.

**Почему dwm.exe в других программах занимает так много?** Dedicated Usage
учитывает все поверхности, которые процесс видит. Здесь используется
Local Usage — память, отнесённая к процессу. Общий объём берётся со счётчика
адаптера, а не суммированием строк. В итог входят только измеренные байты.

**Общая** — системная ОЗУ, доступная GPU. Здесь могут быть закреплённые буферы
хоста и намеренные выделения памяти. Большой объём при нехватке VRAM может
указывать на переход в ОЗУ; один счётчик этого не доказывает.

**Выгр.≈** = max(Total Committed − Local Usage − Non Local Usage, 0).
Это оценка нерезидентной выделенной памяти, а не измеренное число вытеснений
и не доказательство обращения к файлу подкачки.

**Содержимое памяти** определяется косвенно по средам и файлам.
Размеры из журналов описывают выделенные буферы; их текущее размещение
не гарантировано. Перечислить текстуры игры здесь невозможно.

**GPU%** — самый загруженный движок, как в Диспетчере задач. В GeForce/WDDM
ядра CUDA обычно учитываются движком **3D**. Показатель AMD gfx описывает
графический движок и может отличаться по смыслу от GPU utilization NVML.
Vulkan не раскрывает внутренние данные HIP другого процесса.

**Системная панель, RAM и CPU%** — данные ОС (NtQuerySystemInformation
в Windows, psutil в остальных системах). RAM — рабочий набор процесса,
включая общие страницы: сумма строк может превышать занятую память машины.
CPU% — доля мощности всей машины, как в Диспетчере задач. Это системные
показатели; они не прибавляются к памяти GPU. `n/a` означает недоступные
данные, в том числе для строк Windows в WSL; `…` — нужен ещё один замер.
В WSL они описывают виртуальную машину Linux. `--no-system` скрывает их.

**Графики:** по умолчанию видны все восемь, на macOS — доступные два.
В узком окне графики располагаются друг под другом. Меню «Графики» и **c**
переключают все графики или отдельный показатель. Пробел — нет данных;
измеренный ноль отображается у основания. Проценты: 0–100%, температура:
0–120 °C. Масштабы мощности и частоты указаны под графиками.
В невысоком окне можно прокручивать панель и инспектор, сохраняя список процессов.

**Постоянные показатели в Windows:** строка с загрузкой GPU, VRAM и температурой
видна прямо в свободной области основной горизонтальной панели задач, даже
когда основное окно свёрнуто. Перетащите строку левой кнопкой мыши в удобное место.
Положение сохраняется. Правый щелчок по строке или значку → «Показатели на панели
задач» включает или скрывает строку. `--no-taskbar` скрывает её на один запуск.
Если кнопки приложений заняли это место, передвиньте строку; она не резервирует
место в Windows. В полноэкранном приложении и при скрытой панели задач строка скрыта.

**Сворачивание в Windows:** **t** сворачивает обычное консольное окно на панель
задач. Наведите указатель на значок в области уведомлений: там показаны загрузка
GPU, VRAM и температура. В заголовке окна — те же краткие показатели.
Двойной щелчок возвращает обычную консоль; правый щелчок открывает меню.
В Windows Terminal используйте штатную кнопку сворачивания и восстановление
через панель задач: его видимое окно не принадлежит консольному приложению.
Мониторинг продолжается, пока не включена пауза; на паузе значок помечен.
`--no-tray` отключает строку, значок и обновление заголовка. В других ОС значка нет.

Клавиши: **↑/↓** — выбор · **s** — сортировка · **c** — график · **a** — все/основные
· **m** — кратко/подробно · **p** — пауза · **t** — свернуть (Windows)
· **?** — справка · **q** — выход · **Esc** — закрыть справку.
"""


def _add(rows: str):
    for row in rows.strip().splitlines():
        source, target = row.split("\t", 1)
        MESSAGES[source.replace(r"\n", "\n").replace(r"\x20", " ")] = target.replace(r"\n", "\n").replace(r"\x20", " ")


_add(r"""
Compare two --json snapshots, for example before and after loading a model.	Сравнить два снимка --json, например до и после загрузки модели.
earlier --json snapshot	более ранний снимок --json
later --json snapshot	более поздний снимок --json
print the comparison as JSON	вывести сравнение в JSON
gputaskman supports Windows (WDDM), Linux/WSL (NVML, or amdgpu/Intel DRM) and macOS (Apple Silicon) only.	gputaskman поддерживает только Windows (WDDM), Linux/WSL (NVML или amdgpu/Intel DRM) и macOS (Apple Silicon).
Append one --json snapshot per interval to a JSON Lines file, for example to keep what a long run did to the GPU.	Добавлять снимки --json с заданным интервалом в файл JSON Lines, например для записи показателей GPU во время длительной работы.
file to append to, one --json snapshot per line	файл для добавления снимков --json, по одному на строку
GPU to record (default 0; gputaskman --list-gpus shows the indexes)	GPU для записи (по умолчанию 0; индексы: gputaskman --list-gpus)
seconds between samples (default 1)	интервал между замерами в секундах (по умолчанию 1)
stop after this many seconds (default: until Ctrl+C)	остановиться через указанное число секунд (по умолчанию — по Ctrl+C)
WSL: don't read Windows host GPU counters through interop (WSL VM total, Windows apps)	WSL: не читать счётчики GPU хоста Windows через interop (итог ВМ WSL и приложения Windows)
don't sample system RAM/CPU: no system section in the records	не собирать системные ОЗУ/ЦП: раздел system не включается в записи
--gpu must be a non-negative index	--gpu: индекс должен быть неотрицательным
--interval must be a finite number greater than zero	--interval: требуется конечное число больше нуля
--duration must be a finite number greater than zero	--duration: требуется конечное число больше нуля
cannot write {0}: {1}	невозможно записать {0}: {1}
 for {0:g} s	 в течение {0:g} с
Recording GPU {0} every {1:g} s{2} to {3}; press Ctrl+C to stop.	Запись GPU {0} с интервалом {1:g} с{2} в {3}; для остановки нажмите Ctrl+C.
Recorded {0} sample(s) to {1}	Записано замеров: {0}, файл: {1}
What's inside your GPU: per-process VRAM, engines, and LLM weights/KV-cache breakdown.	Что занимает память GPU: VRAM процессов, загрузка движков, веса LLM и KV-кэш.
gputaskman diff BEFORE.json AFTER.json compares two --json snapshots; gputaskman record -o FILE appends one --json snapshot per interval to FILE.	gputaskman diff BEFORE.json AFTER.json сравнивает снимки --json; gputaskman record -o FILE записывает снимки --json с заданным интервалом в FILE.
print one snapshot and exit (no TUI)	вывести один снимок и завершить работу (без интерфейса)
print one JSON snapshot and exit (implies --once)	вывести один снимок JSON и завершить работу (включает --once)
show only this GPU (TUI default: all; one-shot default: 0)	показывать только этот GPU (интерфейс: все; разовый снимок: 0)
list selectable GPU adapters and exit	вывести доступные адаптеры GPU и завершить работу
don't sample system RAM/CPU: no TUI panel or RAM/CPU%% columns, no system section in --once/--json	не собирать системные ОЗУ/ЦП: скрыть панель и столбцы RAM/CPU%%, исключить system из --once/--json
refresh seconds (default 1)	интервал обновления в секундах (по умолчанию 1)
TCC discovery: {0}	Поиск TCC: {0}
interface language (default: ru; or GPUTASKMAN_LANG)	язык интерфейса (по умолчанию ru; также GPUTASKMAN_LANG)
disable the Windows taskbar panel, notification icon and compact window title	отключить строку на панели задач, значок и краткие показатели в заголовке Windows
usage: 	использование:\x20
options	параметры
positional arguments	позиционные аргументы
show this help message and exit	показать справку и завершить работу
error: %s	ошибка: %s
%(prog)s: error: %(message)s\n	%(prog)s: ошибка: %(message)s\n
unrecognized arguments: %s	нераспознанные аргументы: %s
the following arguments are required: %s	обязательные аргументы: %s
argument %s: %s	аргумент %s: %s
invalid choice: %(value)r (choose from %(choices)s)	недопустимое значение: %(value)r (варианты: %(choices)s)
invalid %(type)s value: %(value)r	недопустимое значение типа %(type)s: %(value)r
expected one argument	ожидается один аргумент
""")

_add(r"""
%(prog)s: error: %(message)s	%(prog)s: ошибка: %(message)s
Open monitor	Открыть монитор
Notification icon unavailable: {0}	Значок в области уведомлений недоступен: {0}
Use the terminal's minimize button; readings continue updating.	Сверните окно штатной кнопкой терминала. Показатели продолжат обновляться.
Waiting for GPU readings	Ожидание показателей GPU
[PAUSED] 	[ПАУЗА]\x20
Windows only	Доступно только в Windows
Windows notification area did not respond	Область уведомлений Windows не ответила
Shell_NotifyIconW: notification area unavailable	Shell_NotifyIconW: область уведомлений недоступна
mem-bandwidth	активность памяти
mem	память
encode	кодирование
decode	декодирование
copy	копирование
parent process	родительский процесс
process's own listener	порт самого процесса
max	максимум
decision ({0})	решения ({0})
reported ({0}): {1}	сообщено ({0}): {1}
root	root
""")

_add(r"""
palette	команды
RAM	ОЗУ
CPU%	ЦП%
RAM  	ОЗУ\x20\x20
CPU  	ЦП\x20\x20\x20
Search for commands…	Поиск команд…
Search for themes…	Поиск тем…
No matches found	Совпадений нет
Theme	Тема
Change the current theme	Изменить текущую тему
Quit the application as soon as possible	Завершить работу приложения
Keys	Клавиши
Hide the keys and widget help panel	Скрыть справку о клавишах и элементах интерфейса
Show help for the focused widget and a summary of available keys	Показать справку об интерфейсе и доступных клавишах
Minimize the widget and restore to normal size	Вернуть элемент интерфейса к обычному размеру
Maximize	Развернуть
Maximize the focused widget	Развернуть выбранный элемент интерфейса
Screenshot	Снимок экрана
Save an SVG 'screenshot' of the current screen	Сохранить текущий экран в SVG
Saved screenshot	Снимок экрана сохранён
Saved screenshot to [$text-success]{0!r}	Снимок экрана сохранён в [$text-success]{0!r}
Failed to save screenshot	Не удалось сохранить снимок экрана
argument %(argument_name)s: %(message)s	аргумент %(argument_name)s: %(message)s
\n  loaded: {0} {1} {2} ctx {3}	\n  загружена: {0} {1} {2}, контекст {3}
\n  API-reported model ID: {0}	\n  ID модели по данным API: {0}
""")

_add(r"""
{0}: recording to {1} failed: {2}\n	{0}: ошибка записи в {1}: {2}\n
{0} is not a gputaskman --json snapshot	{0} не является снимком gputaskman --json
snapshots are from different adapters, though both are named {0!r}: {1}	Снимки относятся к разным адаптерам, хотя оба называются {0!r}: {1}
vendor {0:#06x} vs {1:#06x}	производители {0:#06x} и {1:#06x}
adapter key {0:#x} vs {1:#x}	ключи адаптеров {0:#x} и {1:#x}
snapshots are from different GPUs or backends: {0} ({1}) vs {2} ({3})	Снимки относятся к разным GPU или источникам данных: {0} ({1}) и {2} ({3})
TCC compute-only	TCC, только вычисления
reranker	ранжирование
kv	KV-кэш
runtime	среда выполнения
power	мощность
thermal	температура
CPU package	пакет ЦП
vLLM API reported via parent process; model IDs do not prove this process's model residency, weights, KV cache or GPU placement.	Данные API vLLM получены через родительский процесс; ID моделей не подтверждают размещение в памяти этого процесса, объём весов, KV-кэша или размещение на GPU.
vLLM API reported via process's own listener; model IDs do not prove this process's model residency, weights, KV cache or GPU placement.	Данные API vLLM получены через порт самого процесса; ID моделей не подтверждают размещение в памяти этого процесса, объём весов, KV-кэша или размещение на GPU.
Ollama (not matched to this runner): {0} {1} total, {2:.0f}% on GPU, ctx {3}.	Ollama (не сопоставлено с этим рабочим процессом): {0}, всего {1}, на GPU {2:.0f}%, контекст {3}.
KV cache ({0:,} ctx, {1}) est. max	KV-кэш (контекст {0:,}, {1}), оценка максимума
npu-smi not found	npu-smi не найден
npu-smi timed out	Время ожидания npu-smi истекло
npu-smi exited with status {0}	npu-smi завершился с кодом {0}
unrecognized npu-smi process output	Вывод процессов npu-smi не распознан
npu-smi watch starting	Запуск наблюдения npu-smi
npu-smi watch unavailable: npu-smi not found	Наблюдение npu-smi недоступно: npu-smi не найден
npu-smi watch unavailable: {0}	Наблюдение npu-smi недоступно: {0}
npu-smi watch stopped	Наблюдение npu-smi остановлено
npu-smi device memory unavailable: selected chip missing or output unrecognized	Память устройства npu-smi недоступна: выбранный чип отсутствует или вывод не распознан
npu-smi device memory unavailable: {0}	Память устройства npu-smi недоступна: {0}
npu-smi process memory unavailable: {0}	Память процессов npu-smi недоступна: {0}
npu-smi watch data unavailable or stale	Данные наблюдения npu-smi недоступны или устарели
""")


_add(r"""
Quit	Выход
Sort	Сортировка
Chart	График
All/Top	Все/Основные
Slim/Full	Кратко/Подробно
Pause	Пауза
Explain	Справка
Close	Закрыть
Minimize	Свернуть
Sampling GPU counters...	Получение показателей GPU…
Sampling RAM and CPU...	Получение показателей ОЗУ и ЦП…
Charts	Графики
All charts	Все графики
Click to choose · c cycles	Выбор мышью · c — следующий
Process	Процесс
Kind	Тип
Runtime	Среда
Shared	Общая
Paged	Выгр.≈
Copy	Копир.
Video	Видео
What's inside	Содержимое памяти
SM clock	Частота SM
GFX clock	Частота GFX
SM/GFX clock	Частота SM/GFX
Mem activity	Активность памяти
Temperature	Температура
Power	Мощность
Encoder	Кодирование
Decoder	Декодирование
System (OS-reported)	Система (данные ОС)
System (WSL VM, OS-reported)	Система (ВМ WSL, данные ОС)
RAM/CPU unavailable: {0}	ОЗУ/ЦП недоступны: {0}
Full diagnostics	Подробная диагностика
Slim monitoring	Краткий мониторинг
  free {0}	  доступно {0}
pagefile	файл подкачки
swap	подкачка
  {0} threads	  потоков: {0}
/{0} cores	/ядер: {0}
  per-thread	  по потокам
 CPU 60s   	 ЦП 60с\x20\x20\x20
 RAM 60s	 ОЗУ 60с
     top RAM: 	     больше ОЗУ:\x20
     top CPU: 	     больше ЦП:\x20
idle	простой
  unified memory	  объединённая память
   driver {0}	   драйвер {0}
  driver {0}	  драйвер {0}
   [PAUSED]	   [ПАУЗА]
GPU mem 	Память GPU\x20
unattributed {0}  	не распределено {0}\x20\x20
free {0}	свободно {0}
   shared(RAM) {0}	   общая(ОЗУ) {0}
   [bar normalized; labels are raw]	   [шкала нормирована; подписи — исходные значения]
  Mem activity {0}	  активность памяти {0}
  enc {0} dec {1}	  кодир. {0} декод. {1}
  video {0}	  видео {0}
  throttle: 	  ограничения:\x20
Recent samples · blank = no reading	Последние замеры · пробел = нет данных
Recent samples · blank = no reading · power 0–{0:.0f} W · {1} clock 0–{2:.0f} MHz	Последние замеры · пробел = нет данных · мощность 0–{0:.0f} Вт · частота {1} 0–{2:.0f} МГц
sort: {0}  ·  {1} processes	сортировка: {0}  ·  процессы: {1}
all	все
top	основные
paged	выгружено≈
name	имя
Select a process (↑/↓) to see what's inside its GPU memory.  Press ? to learn what's knowable.	Выберите процесс (↑/↓), чтобы посмотреть содержимое памяти GPU. Справка об источниках данных — ?.
host PID 	PID хоста\x20
 - GPU unknown	 - GPU неизвестен
   shared(RAM) 	   общая(ОЗУ)\x20
   paged(est) 	   выгружено(оценка)\x20
   engines 	   движки\x20
   runtime 	   среда\x20
No per-process VRAM: {0}.	Нет данных VRAM по процессам: {0}.
GPU memory n/a: footprint can't read other users' processes without root (or hasn't measured this one yet). GPU% comes from Metal's per-process GPU time.	Память GPU недоступна: footprint не может читать процессы других пользователей без root либо ещё не измерил этот процесс. GPU% рассчитывается по времени GPU процесса в Metal.
Resident VRAM {0}, GTT {1}. 	Резидентная VRAM {0}, GTT {1}.\x20
Engines n/a: this process uses ROCm/HIP (/dev/kfd), whose compute queues the kernel doesn't time per process. The device's AMD gfx load includes its work.	Движки недоступны: процесс использует ROCm/HIP (/dev/kfd); ядро не измеряет время вычислительных очередей по процессам. Его работа входит в общую загрузку AMD gfx.
{0} of {1} committed is not accounted for by local/non-local usage. Estimated non-residency, not proof of eviction or disk paging.	Из {1} выделенной памяти {0} не учтено счётчиками Local/Non Local Usage. Это оценка нерезидентной памяти; она не доказывает вытеснение или подкачку на диск.
Handy selected (settings; not load proof): {0}	В Handy выбрано (настройки; загрузка не подтверждена): {0}
Handy last event: {0} {1} at {2} (runtime-reported history, not current GPU residency)	Последнее событие Handy: {0} {1}, время: {2} (история среды выполнения; текущее размещение в GPU не подтверждено)
Model files, launch flags and runtime buffer logs can span GPUs. The breakdown below is not a measured per-device model placement.	Файлы модели, параметры запуска и журналы буферов могут относиться к нескольким GPU. Разбивка ниже не является измерением размещения модели на выбранном GPU.
(allocated sizes from {0})	(размеры выделенных буферов из {0})
model	модель
arch	архит.
params	параметры
quant	квант.
layers	слои
train ctx	контекст обучения
on GPU (est)	на GPU (оценка)
evidence	основание
file	файл
file: {0}  ({1}, {2}{3})	файл: {0}  ({1}, {2}{3})
server 	сервер\x20
 (parent process; API report, not worker residency)	 (родительский процесс; данные API, размещение в памяти рабочего процесса не подтверждено)
  ctx {0:,}	  контекст {0:,}
  slots busy {0}/{1}	  занято слотов {0}/{1}
  gen {0:.1f} tok/s	  генерация {0:.1f} токен/с
  prompt {0:.0f} tok/s	  обработка запроса {0:.0f} токен/с
No mapped model file was observed. Some runtimes read then close model files. Installed candidates (not proven loaded):	Отображённый в память файл модели не обнаружен. Некоторые среды читают и закрывают файлы моделей. Найденные установленные модели (загрузка не подтверждена):
listening: 	порты прослушивания:\x20
(protected process - run elevated for runtime/file details)	(защищённый процесс — для сведений о среде и файлах нужны права администратора)
decision	принятие решений
decision?	решения?
embedding	эмбеддинги
speech	речь
diffusion	диффузия
AI	ИИ
compute	вычисления
3D/game	3D/игра
video	видео
browser	браузер
app	приложение
compositor	композитор
kernel	ядро
system	система
wsl-vm	ВМ WSL
external	внешний
decision model	модель принятия решений
embedding model	модель эмбеддингов
speech model	речевая модель
vision projector	проектор изображений
LoRA adapter	адаптер LoRA
likely {0} (name)	предположительно {0} (по имени)
confirmed ({0})	подтверждено ({0})
path-arg (not proven loaded)	path-arg (загрузка не подтверждена)
installed, not proven loaded	установлена; загрузка не подтверждена
in use (mapped)	используется (mapped)
launch arg	параметр запуска
reported (api)	сообщено через API
Handy last log ({0}): {1}	Последняя запись Handy ({0}): {1}
Handy selected (not load proof): {0}	В Handy выбрано (загрузка не подтверждена): {0}
parent API	API родительского процесса
reported ({0}): 	сообщено ({0}):\x20
path-arg (unproven): 	path-arg (не подтверждено):\x20
installed (unproven): 	установлена (не подтверждено):\x20
desktop back buffers + composed frames	буферы рабочего стола и скомпонованные кадры
all Linux GPU memory (every WSL distro + WSLg)	вся память GPU в Linux (все дистрибутивы WSL и WSLg)
a process in another PID namespace (host or another container)	процесс из другого пространства PID (хост или другой контейнер)
driver/kernel allocations	память драйвера/ядра
page textures, video, canvas/WebGL	текстуры страниц, видео, canvas/WebGL
video frames (decode/encode)	видеокадры (декодирование/кодирование)
textures, meshes, render targets	текстуры, сетки, цели рендеринга
weights	веса
KV cache	KV-кэш
compute buffers	вычислительные буферы
runtime and other overhead	среда выполнения и прочие расходы
weights ({0} files)	веса (файлов: {0})
KV cache ({0:,} ctx, {1}) est.	KV-кэш (контекст {0:,}, {1}), оценка
KV cache ({0:,} ctx, {1}) est.{2}	KV-кэш (контекст {0:,}, {1}), оценка{2}
compute buffers + runtime	вычислительные буферы и среда выполнения
KV cache + compute + runtime	KV-кэш, вычисления и среда выполнения
runtime-reported	данные среды выполнения
estimated	оценка
unknown	неизвестно
new	новый
gone	исчез
pid	PID
models	модели
added	добавлен
removed	удалён
starting	запуск
ok	работает
off (--no-host)	отключено (--no-host)
off (WSL interop is disabled)	отключено (взаимодействие WSL с Windows запрещено)
off (multi-GPU WSL host matching is not validated; use --gpu N)	отключено (сопоставление нескольких GPU в WSL не проверено; используйте --gpu N)
unmatched (no Windows adapter matches this GPU's usage)	нет соответствия (не найден адаптер Windows с такими показателями GPU)
ok (last Windows sample {0:.0f}s old)	работает (последний замер Windows: {0:.0f} с назад)
VM worker (vmwp.exe)	Рабочий процесс ВМ (vmwp.exe)
WSL VM (vmwp.exe)	ВМ WSL (vmwp.exe)
loaded	загружена
unloaded	выгружена
loading	загрузка
unloading	выгрузка
""")

_add(r"""
vLLM API reported via {0}; model IDs do not prove this process's model residency, weights, KV cache or GPU placement.	Данные API vLLM получены через {0}; ID моделей не подтверждают их размещение в памяти этого процесса, объём весов, KV-кэша или размещение на GPU.
vLLM process signature only; no unambiguous local API association was established.	Обнаружены только признаки процесса vLLM; однозначная связь с локальным API не установлена.
{0} of weights are on the CPU side (partial offload or CPU-mapped).	На стороне ЦП находится {0} весов (частичная выгрузка или отображение в память ЦП).
Weights need ~{0} but the process holds {1} of VRAM: partial offload, paging, or placement on another GPU may explain the gap.	Для весов требуется около {0}, а процессу принадлежит {1} VRAM. Разница может объясняться частичной выгрузкой, подкачкой или размещением на другом GPU.
This process's VRAM total is unknown (WSL charges it to the VM on the Windows host, or the driver hides per-process memory); segments are estimates from the model file and launch flags.	Объём VRAM процесса неизвестен (WSL учитывает его за ВМ на хосте Windows либо драйвер скрывает память процессов). Сегменты — оценки по файлу модели и параметрам запуска.
Sliding-window attention ({0} tokens): most layers keep a short KV window, so the real KV cache is smaller than this estimate.	Скользящее окно внимания ({0} токенов): большинство слоёв хранит короткое окно KV, поэтому реальный KV-кэш меньше этой оценки.
No GGUF mapped. Model may be loaded with --no-mmap or via another format.	Отображённый в память GGUF не найден. Модель могла быть загружена с --no-mmap или в другом формате.
Non-causal encoder ({0}): no KV cache grows with use; VRAM is weights + compute.	Некаузальный энкодер ({0}): растущего KV-кэша нет; VRAM занята весами и вычислениями.
KV cost {0:.0f} KB/token ({1}): 4k ctx = {2}, 32k ctx = {3}.	Объём KV: {0:.0f} КБ/токен ({1}); контекст 4k = {2}, 32k = {3}.
! {0} is in SHARED GPU memory (system RAM over PCIe). May include pinned host buffers; under VRAM pressure, fallback can slow inference.	! {0} находится в ОБЩЕЙ памяти GPU (системная ОЗУ через PCIe). Здесь могут быть закреплённые буферы хоста; при нехватке VRAM использование ОЗУ может замедлять инференс.
Model is mapped but almost nothing is in VRAM: CPU inference, or another GPU.	Модель отображена в память, но почти отсутствует в VRAM: возможен инференс на ЦП или другом GPU.
Ollama{0}: {1} {2} total, {3:.0f}% on GPU, ctx {4}.	Ollama{0}: {1}, всего {2}, на GPU {3:.0f}%, контекст {4}.
LM Studio: {0} {1} {2} ctx {3}.	LM Studio: {0} {1} {2}, контекст {3}.
--gpu {0} is out of range; {1} GPU adapter(s) available	--gpu {0}: индекс вне диапазона; доступно адаптеров GPU: {1}
TCC compute-only GPU: Windows publishes no GPU counters for it, so VRAM and processes come from NVML. Per-process engine load, shared memory and commit are n/a, and per-process VRAM is n/a when the driver does not report it.	GPU TCC только для вычислений: Windows не публикует его счётчики. VRAM и процессы получены из NVML. Загрузка движков, общая и выделенная память по процессам недоступны; VRAM процесса также недоступна, если драйвер её не сообщает.
NVML could not read this TCC GPU; VRAM and processes are unavailable.	NVML не удалось прочитать этот GPU TCC; VRAM и процессы недоступны.
Windows GPU counters could not be matched to the selected adapter; VRAM and processes are unavailable. The adapter may be temporarily unavailable, or GPU-PV VMs may not expose these counters.	Не удалось сопоставить счётчики GPU Windows с выбранным адаптером; VRAM и процессы недоступны. Адаптер может быть временно недоступен; виртуальные машины GPU-PV могут не предоставлять эти счётчики.
Some Windows adapter memory counters are unavailable; GPU-PV VMs may not expose them.	Часть счётчиков памяти адаптера Windows недоступна; виртуальные машины GPU-PV могут не предоставлять их.
NVIDIA telemetry stopped after this GPU was re-enabled; restart gputaskman to restore it.	Телеметрия NVIDIA остановилась после повторного включения GPU; перезапустите gputaskman для её восстановления.
Ascend chip unavailable: npu-smi could not match the selected adapter	Чип Ascend недоступен: npu-smi не удалось сопоставить выбранный адаптер.
Local API on port {0} {1}. Server details are skipped.	Локальный API на порту {0}: {1}. Сведения о сервере пропущены.
rejected the key from the launch arguments	ключ из параметров запуска отклонён
wants an API key, and none was found in the launch arguments (keys from LLAMA_API_KEY or --api-key-file are not read)	требуется ключ API, но в параметрах запуска он не найден (LLAMA_API_KEY и --api-key-file не читаются)
Parent-process API locked (401). Parent probes never send the worker's API key; parent server details are skipped.	API родительского процесса закрыт (401). Ключ рабочего процесса никогда не отправляется родительскому; сведения о сервере пропущены.
Windows adapter discovery unavailable ({0}); retaining last known topology.	Поиск адаптеров Windows недоступен ({0}); сохранена последняя известная конфигурация.
GPU provider cleanup failed: {0}.	Ошибка освобождения ресурсов источника GPU: {0}.
GPU provider initialization failed: {0}.	Ошибка инициализации источника GPU: {0}.
GPU sampling failed: {0}.	Ошибка замера GPU: {0}.
Adapter unavailable; waiting for Windows rediscovery.	Адаптер недоступен; ожидание повторного обнаружения Windows.
NVML is marked unsafe	NVML помечена как небезопасная
NVML returned an invalid PCI address	NVML вернула недопустимый PCI-адрес
NVML is unavailable; TCC GPUs could not be checked.	NVML недоступна; проверить GPU TCC не удалось.
NVML is marked unsafe; TCC GPUs were not checked.	NVML помечена как небезопасная; GPU TCC не проверялись.
NVML initialization failed; TCC GPUs could not be checked.	Ошибка инициализации NVML; проверить GPU TCC не удалось.
NVML encountered an unsafe driver fault; TCC discovery is incomplete.	NVML столкнулась с небезопасной ошибкой драйвера; поиск TCC выполнен не полностью.
NVML TCC discovery was incomplete; one or more devices could not be read.	Поиск TCC через NVML выполнен не полностью: не удалось прочитать одно или несколько устройств.
NVML TCC discovery succeeded; found {0} TCC GPU(s).	Поиск TCC через NVML выполнен; найдено GPU TCC: {0}.
NVML TCC discovery succeeded; no TCC GPUs found.	Поиск TCC через NVML выполнен; GPU TCC не найдены.
 NVML shutdown failed.	 Не удалось завершить работу NVML.
unavailable (powershell.exe not found)	недоступно (powershell.exe не найден)
unavailable ({0})	недоступно ({0})
unavailable (counter stream ended)	недоступно (поток счётчиков завершён)
{0} Linux GPU process(es) without a VM context to match (one may not have created its GPU context yet)	Процессов Linux без подходящего контекста GPU ВМ: {0} (контекст мог ещё не быть создан).
GPU and video load need perf privileges this user lacks (perf_event_paranoid={0})	Для загрузки GPU и видео нужны отсутствующие у пользователя права perf (perf_event_paranoid={0})
no readable supported i915 performance counters for this GPU	нет доступных поддерживаемых счётчиков производительности i915 для этого GPU
performance counters can't attribute load with multiple Intel GPUs	счётчики производительности не позволяют распределить загрузку между несколькими GPU Intel
GPU power needs root (RAPL energy is root-only)	для мощности GPU нужны права root (энергия RAPL доступна только root)
no RAPL uncore zone found	зона RAPL uncore не найдена
Intel device metrics unavailable: 	Показатели устройства Intel недоступны:\x20
. N/A is unknown, not zero.	. N/A означает неизвестное значение, а не ноль.
scan budget reached, {0} processes not rescanned this refresh (last known GPU fds kept)	время сканирования истекло; не проверено процессов: {0} (сохранены последние известные дескрипторы GPU)
{0} processes not inspectable (other users' or kernel threads; sudo includes them)	недоступно для проверки процессов: {0} (другие пользователи или потоки ядра; sudo включает их)
{0} processes whose fds or fdinfo couldn't all be read	процессов с неполностью прочитанными дескрипторами или fdinfo: {0}
DisplayConfig adapter path unavailable (error {0})	Путь адаптера DisplayConfig недоступен (ошибка {0})
PnP device list unavailable (error {0})	Список устройств PnP недоступен (ошибка {0})
PnP adapter interface unavailable (error {0})	Интерфейс адаптера PnP недоступен (ошибка {0})
PnP adapter details unavailable (error {0})	Сведения об адаптере PnP недоступны (ошибка {0})
PnP adapter instance unavailable (error {0})	Экземпляр адаптера PnP недоступен (ошибка {0})
PnP adapter instance is empty	Экземпляр адаптера PnP пуст
Windows PnP identity unavailable ({0}); name and vendor are DXGI-reported.	Идентификация Windows PnP недоступна ({0}); имя и производитель получены из DXGI.
Windows PnP adapter name unavailable; device instance is known.	Имя адаптера Windows PnP недоступно; экземпляр устройства известен.
DXGI adapter discovery failed	Ошибка поиска адаптеров DXGI
DXGI adapter enumeration failed	Ошибка перечисления адаптеров DXGI
DXGI adapter description unavailable	Описание адаптера DXGI недоступно
system RAM/CPU needs 64-bit Python on Windows	Для системных ОЗУ/ЦП в Windows требуется 64-разрядный Python
NtQuerySystemInformation process layout not recognised on this Windows build	Формат списка процессов NtQuerySystemInformation в этой сборке Windows не распознан
""")

_add(r"""
The WSL virtual machine as Windows sees it. Windows charges all GPU memory used by Linux processes in every WSL distro (and WSLg) to vmwp.exe, so this is the exact total. The split across Linux processes is not exposed.	Виртуальная машина WSL с точки зрения Windows. Вся память GPU, используемая процессами Linux во всех дистрибутивах WSL и WSLg, учитывается за vmwp.exe. Это точный общий объём. Разбивка по процессам Linux недоступна.
Desktop Window Manager: its own back buffers and the composed image for each monitor. Windows' 'Dedicated Usage' counter shows dwm as huge (it counts every app surface it can see); this tool prefers Local Usage for attribution. The adapter counter is the total; per-process samples can overlap or differ in timing.	Диспетчер окон рабочего стола: собственные задние буферы и итоговое изображение для каждого монитора. Счётчик Dedicated Usage завышает объём dwm, учитывая все доступные ему поверхности приложений. Здесь для привязки к процессам используется Local Usage. Общий объём даёт счётчик адаптера; замеры процессов могут перекрываться или выполняться в разное время.
Driver/kernel-owned allocations: paging buffers, display scan-out, firmware, scheduler state.	Память драйвера и ядра: буферы подкачки, вывод на дисплей, прошивка, состояние планировщика.
Session/system process holding display and cursor surfaces.	Системный процесс или процесс сеанса, которому принадлежат поверхности дисплея и курсора.
Chromium GPU process: page tiles, images, video frames, canvas, WebGL/WebGPU buffers. Deeper: edge://gpu, or DevTools > Performance with the GPU track.	GPU-процесс Chromium: фрагменты страниц, изображения, видеокадры, canvas, буферы WebGL/WebGPU. Подробнее: edge://gpu или дорожка GPU в DevTools > Performance.
Textures, meshes, render targets and descriptor heaps. Windows only sees opaque allocations; per-resource names need a graphics debugger (PIX on Windows, NVIDIA Nsight Graphics).	Текстуры, сетки, цели рендеринга и кучи дескрипторов. Windows видит только выделенные блоки памяти. Имена отдельных ресурсов доступны в графическом отладчике (PIX для Windows, NVIDIA Nsight Graphics).
Decoded/encoded video frames and codec surfaces (NVIDIA NVDEC/ENC)	Декодированные/кодируемые видеокадры и поверхности кодека (NVIDIA NVDEC/ENC).
Compute allocations are opaque to the OS. Runtime-specific profilers can provide more detail (CUDA: cudaMalloc with Nsight Systems).	ОС не видит содержимое вычислительных блоков памяти. Подробности доступны в профилировщиках среды выполнения (CUDA: cudaMalloc в Nsight Systems).
Model components are identified from launch arguments; GPU allocations cannot be split by component.	Компоненты модели определяются по параметрам запуска; разделить память GPU по компонентам невозможно.
ML runtime allocations (tensors, workspaces). PyTorch: torch.cuda.memory_summary(); ONNX Runtime/DirectML pools are opaque from outside.	Память среды машинного обучения: тензоры и рабочие области. PyTorch: torch.cuda.memory_summary(); внутреннее устройство пулов ONNX Runtime/DirectML извне недоступно.
UI surfaces: swap chains, window back-buffers, cached bitmaps/glyphs.	Поверхности интерфейса: цепочки обмена, задние буферы окон, кэш изображений и символов.
WSL + Windows host counters: 'WSL VM' is the exact total of Linux GPU memory (all distros). Linux rows get their size by matching the VM's GPU contexts (* matched, ? guess). W: rows are Windows processes.	WSL и счётчики хоста Windows: «ВМ WSL» — точный общий объём памяти GPU в Linux (все дистрибутивы). Объёмы строк Linux определяются сопоставлением контекстов GPU виртуальной машины (* — однозначно, ? — предположение). Строки W: — процессы Windows.
 Windows host counters: {0}.	 Счётчики хоста Windows: {0}.
No Linux process in this distro has the GPU open (other WSL distros and Docker Desktop are not visible from here).	В этом дистрибутиве нет процессов Linux с открытым GPU (другие дистрибутивы WSL и Docker Desktop отсюда не видны).
WSL: VRAM and engine load are device-wide. The Windows host charges every Linux process's VRAM to the VM, so rows below are processes using the GPU, not their share of it.	WSL: VRAM и загрузка движков относятся ко всему устройству. Windows учитывает VRAM всех процессов Linux за виртуальной машиной. Строки ниже показывают процессы, использующие GPU; их доли памяти неизвестны.
* matched: the size is exact (Windows' counter for a WSL VM GPU context, or NVML for a host PID seen from a container); which local process owns it is inferred, and only one explanation fits (or every one that fits gives the same size).	* однозначное сопоставление: размер точный (счётчик Windows для контекста GPU ВМ WSL либо NVML для PID хоста, видимого из контейнера). Принадлежность локальному процессу выведена косвенно: подходит только одно объяснение либо все подходящие объяснения дают в точности одинаковый размер.
? order guess: the size is exact but several owners fit, so newer processes are paired with newer GPU contexts / PIDs and outside ones are assumed oldest. Cached foreign-size hints only rank extra candidates; they never prove ownership. Becomes * with evidence, such as a process starting while gputaskman runs.	? предположение по порядку: размер точный, но возможны несколько владельцев. Новые процессы сопоставляются с новыми контекстами GPU/PID; внешние считаются самыми старыми. Кэш размеров лишь упорядочивает кандидатов и не доказывает принадлежность. Метка меняется на * при появлении оснований, например после запуска процесса во время работы gputaskman.
VRAM n/a: this driver's fdinfo has missing or unparseable drm-shared-* stats (older amdgpu kernels, Intel integrated GPUs, xe), so resident memory may include buffers imported from other processes and would be counted once per importer.	VRAM n/a: в fdinfo драйвера отсутствуют или не распознаются показатели drm-shared-* (старые ядра amdgpu, встроенные GPU Intel, xe). Резидентная память может включать импортированные буферы других процессов и учитываться повторно у каждого импортёра.
VRAM and GPU% n/a: this process holds a DRM client other processes hold too (fd passing, fork). fdinfo counts the client once and doesn't say which of them uses it.	VRAM и GPU% n/a: DRM-клиент доступен нескольким процессам (передача дескрипторов, fork). fdinfo учитывает клиента один раз и не указывает, какой процесс его использует.
VRAM n/a: this process's DRM client shares buffers with other clients (dma-buf, e.g. windows and video frames). fdinfo counts shared buffers by where they prefer to live, not where they are, so its private part can't be measured.	VRAM n/a: DRM-клиент процесса разделяет буферы с другими клиентами (dma-buf, например окна и видеокадры). fdinfo учитывает предпочтительное, а не фактическое размещение общих буферов; измерить приватную часть невозможно.
VRAM n/a: this process opened the GPU through /dev/dri/card*, which root-owned services such as logind may also hold where a normal user can't see them; run as root to check.	VRAM n/a: процесс открыл GPU через /dev/dri/card*. Этим же устройством могут владеть невидимые обычному пользователю службы root, например logind. Для проверки запустите от root.
VRAM and GPU% n/a: some of this process's file descriptors or fdinfo couldn't be read.	VRAM и GPU% n/a: часть дескрипторов или fdinfo процесса прочитать не удалось.
VRAM and GPU% n/a this refresh: the scan budget ran out before every process's fds were listed, so another holder of this DRM client can't be ruled out.	VRAM и GPU% n/a в этом замере: время сканирования закончилось до проверки дескрипторов всех процессов. Нельзя исключить другого владельца DRM-клиента.
Resident: every buffer the process can reach, shared ones included. The same buffer can appear under several processes, so these are not added up.	Резидентная память: все доступные процессу буферы, включая общие. Один буфер может быть указан у нескольких процессов, поэтому эти значения не суммируются.
macOS: unified memory, so GPU memory is part of system RAM. Rows show each process's graphics footprint; other users' processes (WindowServer, daemons) show GPU% only, since reading their memory needs root.	macOS: память объединённая, память GPU входит в системную ОЗУ. Строки показывают графическую память процессов. Для процессов других пользователей (WindowServer, службы) доступна только GPU%: чтение их памяти требует root.
Intel VRAM used is the sum of memory resident in the processes gputaskman can inspect (the driver has no device-wide counter): local memory on a discrete card, system memory on an integrated GPU. Buffers shared between processes can be counted twice, and kernel-owned memory and processes it can't inspect are missing. The total is the size of the card's local-memory PCI window, an estimate that can differ from its memory (0 on an integrated GPU, whose total is unknown), so free memory is not shown.	Занятая VRAM Intel — сумма резидентной памяти доступных для проверки процессов: локальной на дискретной карте, системной на встроенном GPU. Счётчика всего устройства у драйвера нет. Общие буферы могут учитываться дважды; память ядра и недоступных процессов не учтена. Общий объём — размер PCI-окна локальной памяти карты, то есть оценка, которая может отличаться от реального объёма (0 для встроенного GPU означает неизвестный объём). Свободная память не показывается.
GPU memory {0} allocated / {1} unified	Память GPU: выделено {0} / объединённая {1}
   shared (in system RAM) {0}	   общая (в системной ОЗУ) {0}
DRM process scan: {0}.	Сканирование процессов DRM: {0}.
Linux VRAM split: {0}	Распределение VRAM в Linux: {0}
== {0} ({1}) Handy selected (settings, not load proof): {2}	== {0} ({1}) В Handy выбрано (настройки; загрузка не подтверждена): {2}
   last Handy event: {0} {1} at {2} (runtime-reported history, not current GPU residency)	   Последнее событие Handy: {0} {1}, время: {2} (история среды выполнения; текущее размещение в GPU не подтверждено)
   file: {0}  ({1})  evidence: {2}	   файл: {0}  ({1})  основание: {2}
   {0}: {1}  [{2}, {3} params, {4}, {5} layers, ctx_train {6}]  {7}  evidence: {8}	   {0}: {1}  [{2}, параметров: {3}, {4}, слоёв: {5}, контекст обучения: {6}]  {7}  основание: {8}
   file: {0}  ({1}{2})  evidence: {3}	   файл: {0}  ({1}{2})  основание: {3}
   server: {0} {1} ctx={2} slots={3}/{4}	   сервер: {0} {1} контекст={2} слоты={3}/{4}
 gen={0:.1f} tok/s	 генерация={0:.1f} токен/с
   server source: parent process (API report, not worker residency)	   источник сервера: родительский процесс (данные API; размещение в памяти рабочего процесса не подтверждено)
   API-reported model ID: {0}	   ID модели по данным API: {0}
   exact sizes from log: {0}	   точные размеры выделенных буферов из журнала: {0}
System RAM/CPU unavailable: {0}	Системные ОЗУ/ЦП недоступны: {0}
System RAM and CPU (OS-reported; not GPU memory)	Системные ОЗУ и ЦП (данные ОС; не память GPU)
CPU {0:.0f}%  {1} threads	ЦП {0:.0f}%  потоков: {1}
 / {0} cores	 / ядер: {0}
RAM {0} / {1}  available {2}   swap/pagefile {3} / {4}	ОЗУ {0} / {1}  доступно {2}   подкачка {3} / {4}
Top processes by RAM, whole machine:	Процессы с наибольшим потреблением ОЗУ, вся система:
({0} processes not readable without admin)	(процессов, недоступных без прав администратора: {0})
PROCESS	ПРОЦЕСС
KIND	ТИП
RUNTIME	СРЕДА
SHARED	ОБЩАЯ
PAGED	ВЫГР.≈
INSIDE	СОДЕРЖИМОЕ
PRIVATE	ЧАСТНАЯ
BEFORE	ДО
AFTER	ПОСЛЕ
CHANGE	ИЗМЕНЕНИЕ
MATCH	СОПОСТАВЛ.
dedicated used	занятая VRAM
shared used	занятая общая
attributed to processes	отнесено к процессам
unattributed	не распределено
, {0} s apart	, интервал {0} с
unknown GPU	неизвестный GPU
Measured counters are compared directly. Runtime-reported and estimated values are listed	Измеренные показатели сравниваются напрямую. Данные среды выполнения и оценки указаны
separately and are not added to measured totals.	отдельно и не прибавляются к измеренным итогам.
Warning: {0}	Внимание: {0}
DEVICE (measured)	УСТРОЙСТВО (измерено)
PROCESSES (dedicated)	ПРОЦЕССЫ (VRAM)
SHARED +/-	ОБЩАЯ +/-
not listed	нет в списке
(no process changed by 16 MB or more)	(ни у одного процесса изменение не достигло 16 МБ)
{0} process(es) changed by less than 16 MB are not shown; --json lists all.	Не показаны процессы с изменением менее 16 МБ: {0}; --json выводит все процессы.
  inferred VRAM (WSL context match)  {0} -> {1}  {2}	  косвенно определённая VRAM (контексты WSL)  {0} -> {1}  {2}
  context length  {0} -> {1}	  длина контекста  {0} -> {1}
  model file {0}: {1} [{2}]	  файл модели {0}: {1} [{2}]
Model files show what a process opened or named, not what is resident in GPU memory.	Файлы моделей показывают, что процесс открыл или указал при запуске. Это не подтверждает размещение в памяти GPU.
'not listed' means the process was absent from that snapshot: not running, or using too little GPU to be listed.	«нет в списке» означает отсутствие процесса в снимке: он не работал либо использовал слишком мало ресурсов GPU для включения в список.
Their adapter keys differ and no PCI address is available to settle it. Windows changes those keys at boot, so this may be one GPU or two of the same model; they are matched by GPU name and backend only.	Ключи адаптеров различаются; PCI-адрес для проверки недоступен. Windows меняет ключи при загрузке, поэтому это может быть один GPU или две карты одной модели. Сопоставление выполнено только по имени GPU и источнику данных.
These snapshots do not both say which adapter they came from (an older gputaskman, or no adapter details), so they are matched by GPU name and backend only.	Не в обоих снимках указан адаптер (старая версия gputaskman или нет сведений об адаптере). Сопоставление выполнено только по имени GPU и источнику данных.
""")
