import argparse
import sys
import xml.etree.ElementTree as ET


class BaseLoader:
    def __init__(self, sentences):
        self.sentences = sentences

    @staticmethod
    def handle_text(text):
        print('text: {}'.format(text))

    @staticmethod
    def handle_aspect_term(attrs):
        term = attrs['term']
        polarity = attrs['polarity']
        from_ = attrs['from']
        to = attrs['to']
        print('aspectTerm: {}'.format(attrs))

    @staticmethod
    def handle_aspect_category(attrs):
        category = attrs['category']
        polarity = attrs['polarity']
        print('aspectCategory: {}'.format(attrs))

    def parse_text(self, text):
        self.handle_text(text.text)

    def parse_aspect_terms(self, aspect_terms):
        for aspect_term in aspect_terms:
            self.parse_aspect_term(aspect_term)

    def parse_aspect_term(self, aspect_term):
        attrs = aspect_term.attrib
        self.handle_aspect_term(attrs)

    def parse_aspect_categories(self, aspect_categories):
        for aspect_category in aspect_categories:
            self.parse_aspect_category(aspect_category)

    def parse_aspect_category(self, aspect_category):
        attrs = aspect_category.attrib
        self.handle_aspect_category(attrs)

    def parse(self):
        for sentence in self.sentences:
            self.parse_text(sentence[0])
            if len(sentence) > 1:
                child = sentence[1]
                if child.tag == 'aspectTerms':
                    self.parse_aspect_terms(child)
                elif child.tag == 'aspectCategories':
                    self.parse_aspect_categories(child)
            if len(sentence) > 2:
                self.parse_aspect_categories(sentence[2])


def main(loader_class):
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", default='restaurants-trial.xml', help='Input data file in xml format')
    parser.add_argument("-output", default='restaurants-trial.txt', help='Output data file in txt format')
    args = parser.parse_args()

    orig_stdout = sys.stdout
    out_file = open(args.output, 'w')
    sys.stdout = out_file

    sentences = ET.parse(args.input).getroot()

    loader = loader_class(sentences)
    loader.parse()

    sys.stdout = orig_stdout
    out_file.close()


if __name__ == '__main__':
    main(BaseLoader)
