package com.gengoai.apollo.ml.clustering.flat;

import com.gengoai.Validation;
import com.gengoai.math.Optimum;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.ml.clustering.Clustering;
import com.gengoai.apollo.stat.measure.Measure;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

import java.util.Iterator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * @author David B. Bracewell
 */
public class GMM extends Clustering {
   private static final long serialVersionUID = 1L;
   List<MultivariateNormalDistribution> components;

   public GMM(Clusterer<?> clusterer, Measure measure) {
      super(clusterer, measure);
   }

   @Override
   public Cluster get(int index) {
      Validation.checkElementIndex(index, components.size());
      Cluster c = new Cluster();
      c.setCentroid(NDArrayFactory.wrap(components.get(index).sample()));
      return c;
   }

   @Override
   public int hardCluster(Instance instance) {
      return Optimum.MAXIMUM.optimumIndex(softCluster(instance));
   }

   @Override
   public Iterator<Cluster> iterator() {
      return IntStream.range(0, components.size())
                      .mapToObj(this::get)
                      .iterator();
   }

   @Override
   public int size() {
      return components.size();
   }

   @Override
   public double[] softCluster(Instance instance) {
      double[] d = new double[components.size()];
      double[] in = getPreprocessors().apply(instance).toVector(getEncoderPair()).toArray();
      for (int i = 0; i < components.size(); i++) {
         d[i] = components.get(i).density(in);
      }
      return d;
   }

}// END OF GMM
