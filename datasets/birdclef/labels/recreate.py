#!/usr/bin/env python
# Collect labels and metadata.
import sys
import os
import io
from collections import defaultdict, Counter


def find_xml_files(dir):
    """Yields the file names of all XML files in a given directory tree."""
    for dirpath, dirnames, filenames in os.walk(dir, followlinks=True):
        for fn in filenames:
            if fn.endswith('.xml'):
                yield os.path.join(dirpath, fn)


def repair_time(time):
    time = time.lower()
    for ignore in ('hs', 'h', 's', 'min', 'm', '.m.'):
        if time.endswith(ignore):
            time = time[:-len(ignore)]
    for ignore in ('~',):
        if time.startswith(ignore):
            time = time[len(ignore):]
    pm = False
    if time.endswith('a'):
        time = time[:-1]
    if time.endswith('p'):
        pm = True
        time = time[:-1]
    for sep in ('.:', '.', '-', ',', 'h', '_'):
        time = time.replace(sep, ':')
    if time.endswith(' 00') and time[-4] != ':':
        time = time[:-3] + ':00'
    time = list(map(int, time.split(':')))
    if pm:
        time[0] += 12
    if len(time) < 3:
        time.extend([0] * (3 - len(time)))
    return '%02d:%02d:%02d' % tuple(time)


def repair_elevation(elevation):
    for ignore in ('&lt;b&gt;', 'c ', 'c.', 'ca.', 'approx.', 'aprox.', 'about'):
        if elevation.startswith(ignore):
            elevation = elevation[len(ignore):]
    for ignore in ('&lt;/b&gt;', '?', 'm', 'metros', 'approximately'):
        if elevation.endswith(ignore):
            elevation = elevation[:-len(ignore)]
    for upto in ('&lt;', 'to '):
        if elevation.startswith(upto):
            elevation = '0 - ' + elevation[len(upto):]
    elevation = elevation.replace(' a ', '-')  # 123 a 456 >= 123-456
    elevation = elevation.replace('.', '')  # hope to ignore thousands separator
    if '-' in elevation:
        return '-'.join(str(int(e)) for e in elevation.split('-'))
    else:
        return str(int(elevation))


def parse_xml_file(filename):
    """Returns a dictionary with the class_id, genus, species, bg_species."""
    def strip_tag(s):
        return s.split('>', 1)[1].split('<', 1)[0]

    mapping = {"ClassId": "class_id",
               "Genus": "genus",
               "Species": "species",
               "BackgroundSpecies": "bg_species",
               "Date": "date",
               "Time": "time",
               "Latitude": "latitude",
               "Longitude": "longitude",
               "Elevation": "elevation",
               }

    result = {}
    with io.open(filename) as f:
        for line in f:
            line = line.lstrip()
            for k, v in mapping.items():
                if line.startswith('<%s>' % k):
                    result[v] = strip_tag(line)

    # parseability checks and normalization
    try:
        list(map(int, result.get('date', '0').split('-')))
    except ValueError:
        del result['date']
    if 'time' in result:
        try:
            result['time'] = repair_time(result['time'])
        except ValueError:
            del result['time']
    try:
        float(result.get('latitude', 0))
    except ValueError:
        del result['latitude']
    try:
        float(result.get('longitude', 0))
    except ValueError:
        del result['longitude']
    if 'elevation' in result:
        try:
            result['elevation'] = repair_elevation(result['elevation'])
        except ValueError as e:
            del result['elevation']
    return result


def main():
    write = 'wb' if sys.version_info[0] == 2 else 'w'

    # slurp in all the XML files
    xmldir, xmldir_test, xmldir_scapes, outdir = sys.argv[1:]
    data = {}
    for fn in find_xml_files(xmldir):
        key = os.path.join('train', fn[len(xmldir):].lstrip('/')[:-3] + 'wav')
        data[key] = parse_xml_file(fn)
    data_test = {}
    for fn in find_xml_files(xmldir_test):
        key = os.path.join('test',
                           fn[len(xmldir_test):].lstrip('/')[:-3] + 'wav')
        data_test[key] = parse_xml_file(fn)
    data_scapes = {}
    for fn in find_xml_files(xmldir_scapes):
        key = os.path.join('soundscapes',
                           fn[len(xmldir_scapes):].lstrip('/')[:-3].replace(
                                   '/xml', '') + 'wav')
        data_scapes[key] = parse_xml_file(fn)

    # shortcut during development
#    import cPickle as pickle
#    with open('data.pkl', 'wb') as f:
#        pickle.dump(data, f, -1)
#    with open('data.pkl', 'rb') as f:
#        data = pickle.load(f)

    # collect the set of class IDs
    labelset = sorted(set(d['class_id'] for d in data.values()))
    with io.open(os.path.join(outdir, 'labelset'), write) as f:
        f.writelines('%s\n' % label for label in labelset)

    print("Figuring out the mapping between latin names and class IDs...")
    # We use the foreground species annotations to figure out which latin name
    # corresponds to which class ID, so we can use it to convert the background
    # species labels (which use latin names). Unfortunately, sometimes the
    # foreground species from the XML does not match the class ID from the XML,
    # so we count associations to figure out which is used the most.

    # map class IDs to latin names
    id_to_latin = defaultdict(Counter)
    for d in data.values():
        id_to_latin[d['class_id']].update([d['genus'] + " " + d['species']])
    for k, v in sorted(id_to_latin.items()):
        if len(v) > 1:
            print("- %s mapped to %s" %
                    (k, " and ".join("%s (%dx)" % x for x in v.most_common())))
    id_to_latin = {k: v.most_common(1)[0][0] for k, v in id_to_latin.items()}
    with io.open(os.path.join(outdir, 'labelset_latin'), write) as f:
        f.writelines("%s\n" % id_to_latin[class_id] for class_id in labelset)

    # map latin names to class IDs
    latin_to_id = defaultdict(Counter)
    for d in data.values():
        latin_to_id[d['genus'] + " " + d['species']].update([d['class_id']])
    for k, v in sorted(latin_to_id.items()):
        if len(v) > 1:
            print("- %s mapped to %s" %
                    (k, " and ".join("%s (%dx)" % x for x in v.most_common())))
    latin_to_id = {k: v.most_common(1)[0][0] for k, v in latin_to_id.items()}

    print("Inconsistencies were resolved by using the most-used associations.")

    # map class IDs to integers
    id_to_int = dict((label, idx) for idx, label in enumerate(labelset))

    # write out the foreground species
    with io.open(os.path.join(outdir, 'fg.tsv'), write) as f:
        f.writelines("%s\t%d\n" % (fn, id_to_int[d['class_id']])
                     for fn, d in data.items())

    # write out the background species
    with io.open(os.path.join(outdir, 'bg.tsv'), write) as f:
        f.writelines("%s\t%s\n" %
                (fn, ",".join(str(id_to_int[latin_to_id[latin]])
                              for latin in d['bg_species'].split(',')
                              if latin in latin_to_id)
                     if 'bg_species' in d else "")
                     for fn, d in data.items())

    # write out label subsets for the different challenge editions
    fg2014 = set(d['class_id'] for fn, d in data.items() if 'CLEF2014' in fn)
    fg2015 = set(d['class_id'] for fn, d in data.items() if 'CLEF2015' in fn)
    fg2017 = set(d['class_id'] for fn, d in data.items() if 'CLEF2017' in fn)
    with io.open(os.path.join(outdir, 'labelset_2014'), write) as f:
        f.writelines('%s\n' % label for label in sorted(fg2014))
    with io.open(os.path.join(outdir, 'labelset_2015'), write) as f:
        f.writelines('%s\n' % label for label in sorted(fg2014 | fg2015))
    with io.open(os.path.join(outdir, 'labelset_2017'), write) as f:
        f.writelines('%s\n' % label for label in sorted(fg2017))

    # write out other metadata
    data.update(data_test)
    data.update(data_scapes)
    with io.open(os.path.join(outdir, 'meta.tsv'), write) as f:
        f.writelines('%s\t%s\t%s\t%s\t%s\t%s\n' %
                (fn, d.get('date', ''), d.get('time', ''),
                 d.get('latitude', ''), d.get('longitude', ''),
                 d.get('elevation', ''))
                for fn, d in data.items())


if __name__ == "__main__":
    main()
