from sys import argv

from .constants import LANGUAGES

SPLITS = ['original', 'custom']


def usage() -> None:
    print('\nUsage: ' + argv[0] + ' ' + '<language> [data split]\n')
    print(argv[0] + ': Program name.')
    print('<language>: The language for which the program is called.')
    print('            <language> can be: ', end='')
    for lang in LANGUAGES[:-1]:
        print(lang, end='/')
    print(LANGUAGES[-1] + '\n')
    print('[data split]: The data split to use.')
    print('              This can be <original> or <custom> split.')
    print('              Optional argument. <original> split by default.')


def parse_program_args() -> tuple[str, str] | None:
    if len(argv) >= 2 and argv[1] in LANGUAGES:
        if len(argv) == 2:
            return argv[1], 'original'
        if len(argv) == 3 and argv[2] in SPLITS:
            return argv[1], argv[2]
    usage()
    raise Exception('Invalid usage')
