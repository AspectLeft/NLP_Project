import argparse
import sys
import xml.etree.ElementTree as ET


def parse_text(text):
    print('text: {}'.format(text.text))


def parse_aspect_terms(aspect_terms):
    for aspect_term in aspect_terms:
        parse_aspect_term(aspect_term)


def parse_aspect_term(aspect_term):
    attrs = aspect_term.attrib
    term = attrs['term']
    polarity = attrs['polarity']
    from_ = attrs['from']
    to = attrs['to']
    print('aspectTerm: {}'.format(attrs))


def parse_aspect_categories(aspect_categories):
    for aspect_category in aspect_categories:
        parse_aspect_category(aspect_category)


def parse_aspect_category(aspect_category):
    attrs = aspect_category.attrib
    category = attrs['category']
    polarity = attrs['polarity']
    print('aspectCategory: {}'.format(attrs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", default='restaurants-trial.xml', help='Input data file in xml format')
    parser.add_argument("-output", default='restaurants-trial.txt', help='Output data file in txt format')
    args = parser.parse_args()

    orig_stdout = sys.stdout
    out_file = open(args.output, 'w')
    sys.stdout = out_file

    sentences = ET.parse(args.input).getroot()


    for sentence in sentences:
        parse_text(sentence[0])
        if len(sentence) > 1:
            child = sentence[1]
            if child.tag == 'aspectTerms':
                parse_aspect_terms(child)
            elif child.tag == 'aspectCategories':
                parse_aspect_categories(child)
        if len(sentence) > 2:
            parse_aspect_categories(sentence[2])

    sys.stdout = orig_stdout
    out_file.close()


if __name__ == '__main__':
    main()
