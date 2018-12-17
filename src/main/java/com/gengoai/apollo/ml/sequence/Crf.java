package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.tuple.Tuple2;
import com.github.jcrfsuite.CrfTagger;
import com.github.jcrfsuite.util.Pair;
import third_party.org.chokkan.crfsuite.*;

import java.io.IOException;
import java.io.Serializable;
import java.util.Base64;
import java.util.List;

import static com.gengoai.tuple.Tuples.$;

/**
 * The type Crf.
 *
 * @author David B. Bracewell
 */
public class Crf implements SequenceLabeler, Serializable {
   private static final long serialVersionUID = 1L;
   private String modelFile;
   private volatile CrfTagger tagger;
   private int numberOfFeatures = 0;


   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters parameters) {
      Parameters fitParameters = Cast.as(parameters, Parameters.class);
      CrfSuiteLoader.INSTANCE.load();
      Trainer trainer = new Trainer();
      this.numberOfFeatures = fitParameters.numFeatures;
      dataSupplier.get().forEachLocal(sequence -> sequence.sliceStream()
                                                          .forEach(p -> {
                                                             Tuple2<ItemSequence, StringList> seq = toItemSequence(p.v2,
                                                                                                                   true);
                                                             trainer.append(seq.v1, seq.v2, 0);
                                                          }));
      trainer.select(fitParameters.crfSolver.parameterSetting, "crf1d");
      trainer.set("max_iterations", Integer.toString(fitParameters.maxIterations));
      trainer.set("c2", Double.toString(fitParameters.c2));
      modelFile = Resources.temporaryFile().asFile().get().getAbsolutePath();
      trainer.train(modelFile, -1);
      trainer.clear();
      tagger = new CrfTagger(modelFile);
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Crf.Parameters();
   }


   private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException {
      CrfSuiteLoader.INSTANCE.load();
      Resource tmp = Resources.temporaryFile();
      int length = stream.readInt();
      byte[] bytes = new byte[length];
      stream.readFully(bytes);
      tmp.write(Base64.getDecoder().decode(bytes));
      this.modelFile = tmp.asFile().get().getAbsolutePath();
      this.tagger = new CrfTagger(modelFile);
   }

   private Tuple2<ItemSequence, StringList> toItemSequence(NDArray sequence, boolean includeLabel) {
      ItemSequence seq = new ItemSequence();
      StringList labels = new StringList();
      NDArray y = null;
      if (includeLabel) {
         y = sequence.getLabelAsNDArray();
         if (y.isMatrix()) {
            y = y.argMax(Axis.ROW);
         }
      }
      for (int row = 0; row < sequence.numRows(); row++) {
         Item item = new Item();
         sequence.sparseRowIterator(row)
                 .forEachRemaining(e -> item.add(new Attribute(
                    Integer.toString(e.getColumn()),
                    e.getValue()
                 )));
         if (includeLabel) {
            labels.add(Integer.toString((int) y.get(row)));
         }
         seq.add(item);
      }
      return $(seq, labels);
   }

   @Override
   public NDArray estimate(NDArray sequence) {
      CrfSuiteLoader.INSTANCE.load();
      sequence.sliceStream()
              .forEach(p -> {
                 Tuple2<ItemSequence, StringList> seq = toItemSequence(p.v2, false);
                 List<Pair<String, Double>> tags = tagger.tag(seq.v1);
                 NDArray labels = NDArrayFactory.DENSE.zeros(1, tags.size());
                 for (int i = 0; i < tags.size(); i++) {
                    labels.set(i, Integer.parseInt(tags.get(i).first));
                 }
                 p.v2.setPredicted(labels);
              });
      return sequence;
   }

   private void writeObject(java.io.ObjectOutputStream stream) throws IOException {
      byte[] modelBytes = Base64.getEncoder().encode(Resources.from(modelFile).readBytes());
      stream.writeInt(modelBytes.length);
      stream.write(modelBytes);
   }

   @Override
   public int getNumberOfFeatures() {
      return numberOfFeatures;
   }

   @Override
   public int getNumberOfLabels() {
      return tagger.getlabels().size();
   }

   /**
    * The type Parameters.
    */
   public static class Parameters extends FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The C 2.
       */
      public double c2 = 1.0;
      /**
       * The Max iterations.
       */
      public int maxIterations = 50;
      /**
       * The Solver.
       */
      public CrfSolver crfSolver = CrfSolver.LBFGS;
   }

}//END OF Crf
