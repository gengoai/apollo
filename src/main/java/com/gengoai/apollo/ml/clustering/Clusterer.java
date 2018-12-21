package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.Streams;
import com.gengoai.tuple.Tuple2;

import java.util.List;
import java.util.stream.Collectors;

import static com.gengoai.tuple.Tuples.$;

/**
 * The type Clusterer.
 *
 * @author David B. Bracewell
 */
public abstract class Clusterer extends Model implements Iterable<Cluster> {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Clusterer.
    *
    * @param preprocessors the preprocessors
    */
   public Clusterer(Preprocessor... preprocessors) {
      super(IndexVectorizer.labelVectorizer(),
            IndexVectorizer.featureVectorizer(),
            preprocessors);
   }

   /**
    * Instantiates a new Clusterer.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public Clusterer(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }


   /**
    * Instantiates a new Clusterer.
    *
    * @param labelVectorizer   the label vectorizer
    * @param featureVectorizer the feature vectorizer
    * @param preprocessors     the preprocessors
    */
   public Clusterer(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }


   /**
    * Cluster nd array.
    *
    * @param data the data
    * @return the nd array
    */
   public NDArray cluster(Example data) {
      return softCluster(data);
   }

   /**
    * Cluster nd array.
    *
    * @param data the data
    * @return the nd array
    */
   public NDArray cluster(NDArray data) {
      return softCluster(data);
   }

   /**
    * Gets labels.
    *
    * @param id the id
    * @return the labels
    */
   public List<String> getLabels(int id) {
      return getCluster(id)
                .getPoints()
                .stream()
                .map(n -> getLabelVectorizer().decode(n.getLabelAsDouble()).toString())
                .collect(Collectors.toList());
   }

   /**
    * Gets cluster.
    *
    * @param id the id
    * @return the cluster
    */
   public abstract Cluster getCluster(int id);

   /**
    * Gets measure.
    *
    * @return the measure
    */
   public abstract Measure getMeasure();

   /**
    * Gets root.
    *
    * @return the root
    */
   public abstract Cluster getRoot();


   /**
    * Hard cluster int.
    *
    * @param example the example
    * @return the int
    */
   public int hardCluster(Example example) {
      return hardCluster(encodeAndPreprocess(example));
   }

   /**
    * Performs a hard clustering, which determines the single cluster the given instance belongs to
    *
    * @param vector the vector
    * @return the index of the cluster that the instance belongs to
    */
   public int hardCluster(NDArray vector) {
      return getMeasure().getOptimum()
                         .optimum(Streams.asParallelStream(this)
                                         .map(c -> $(c.getId(), getMeasure().calculate(vector, c.getCentroid()))),
                                  Tuple2::getV2)
                         .map(Tuple2::getKey)
                         .orElse(-1);
   }

   /**
    * Checks if the clustering is flat
    *
    * @return True if flat, False otherwise
    */
   public boolean isFlat() {
      return false;
   }

   /**
    * Checks if the clustering is hierarchical
    *
    * @return True if hierarchical, False otherwise
    */
   public boolean isHierarchical() {
      return false;
   }

   /**
    * The number of clusters
    *
    * @return the number of clusters
    */
   public abstract int size();


   /**
    * Soft cluster nd array.
    *
    * @param example the example
    * @return the nd array
    */
   public NDArray softCluster(Example example) {
      return softCluster(encodeAndPreprocess(example));
   }

   /**
    * Performs a soft clustering, which provides a membership probability of the given instance to the clusters
    *
    * @param instance the instance
    * @return membership probability of the given instance to the clusters
    */
   public NDArray softCluster(NDArray instance) {
      NDArray distances = NDArrayFactory.DENSE.zeros(size()).fill(Double.POSITIVE_INFINITY);
      int assignment = hardCluster(instance);
      if (assignment >= 0) {
         distances.set(assignment, getMeasure().calculate(getCluster(assignment).getCentroid(), instance));
      }
      return distances;
   }


}//END OF Clusterer
