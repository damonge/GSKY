class SampleFileUtil(object):
    """
    Util for handling sample files

    :param filePrefix: the prefix to use
    :param master: True if the sampler instance is the master
    :param  reuseBurnin: True if the burn in data from a previous run should be used

    """

    def __init__(self, filePrefix):

        self.filePrefix = filePrefix

        self.samplesFile = open(self.filePrefix + '.out', "w")
        self.probFile = open(self.filePrefix + 'prob.out', "w")

    def persistSamplingValues(self, pos, prob):
        self.persistValues(self.samplesFile, self.probFile, pos, prob)

    def persistValues(self, posFile, probFile, pos, prob):
        """
        Writes the walker positions and the likelihood to the disk
        """
        posFile.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
        posFile.write("\n")
        posFile.flush()

        probFile.write("\n".join([str(p) for p in prob]))
        probFile.write("\n")
        probFile.flush();

    def close(self):
        self.samplesFile.close()
        self.probFile.close()

    def __str__(self, *args, **kwargs):
        return "SampleFileUtil"