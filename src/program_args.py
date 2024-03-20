from sys import argv


LANGUAGES = ['afr', 'eng', 'esp', 'hin', 'mar', 'pan']


def usage() -> None:
    print('\nUsage: ' + argv[0] + ' ' + 'language\n')
    print(argv[0] + ': program name')
    print('language: tha language for which the program is called')
    print('          language can be: ', end='')
    for lang in LANGUAGES[:-1]:
        print(lang, end='/')
    print(LANGUAGES[-1] + '\n')


def parse_program_args() -> None:
    if len(argv) == 2 and argv[1] in LANGUAGES:
        return
    usage()
    raise Exception('Invalid usage')
