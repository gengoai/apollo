package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.Interner;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.collect.Lists;
import com.davidbracewell.string.StringUtils;

import java.io.Serializable;
import java.util.*;

/**
 * The type Transition features.
 *
 * @author David B. Bracewell
 */
public class TransitionFeatures implements Serializable {
   /**
    * The constant FIRST_ORDER.
    */
   public static final TransitionFeatures FIRST_ORDER = new TransitionFeatures("-1");
   /**
    * The constant SECOND_ORDER.
    */
   public static final TransitionFeatures SECOND_ORDER = new TransitionFeatures("-2,-1");
   private static final long serialVersionUID = 1L;
   private static final Interner<String> INTERNER = new Interner<>();
   private final int[][] featureTemplates;
   private int historySize;

   /**
    * Instantiates a new Transition features.
    *
    * @param templates the templates
    */
   public TransitionFeatures(String... templates) {
      this.featureTemplates = new int[Preconditions.checkNotNull(templates).length][];
      for (int i = 0; i < templates.length; i++) {
         List<String> temp = StringUtils.split(templates[i], ',');
         this.featureTemplates[i] = new int[temp.size()];
         for (int j = 0; j < temp.size(); j++) {
            this.featureTemplates[i][j] = Integer.parseInt(temp.get(j));
            this.historySize = Math.max(this.historySize, Math.abs(this.featureTemplates[i][j]));
         }
      }
   }


   /**
    * To instances list.
    *
    * @param sequence the sequence
    * @return the list
    */
   public List<Instance> toInstances(Sequence sequence) {
      Context<Instance> itr = sequence.iterator();
      List<Instance> instances = new ArrayList<>();
      while (itr.hasNext()) {
         Instance instance = itr.next();
         List<Feature> features = Lists.newArrayList(instance);
         Iterator<String> transitions = extract(itr);
         while (transitions.hasNext()) {
            features.add(Feature.TRUE(transitions.next()));
         }
         instances.add(Instance.create(features, instance.getLabel()));
      }
      return instances;
   }

   /**
    * Fit transitions features.
    *
    * @param dataset the dataset
    */
   public void fitTransitionsFeatures(Dataset<Sequence> dataset) {
      dataset.getFeatureEncoder().fit(
         dataset.stream()
                .flatMap(sequence -> {
                            Context<Instance> ci = sequence.iterator();
                            Set<String> features = new HashSet<>();
                            while (ci.hasNext()) {
                               ci.next();
                               Iterator<String> itr = extract(ci);
                               while (itr.hasNext()) {
                                  features.add(itr.next());
                               }
                            }
                            return features.stream();
                         }
                        )
                                     );
   }

   /**
    * Gets history size.
    *
    * @return the history size
    */
   public int getHistorySize() {
      return historySize;
   }


   /**
    * Extract iterator.
    *
    * @param iterator the iterator
    * @return the iterator
    */
   public Iterator<String> extract(final Context<Instance> iterator) {
      return new Iterator<String>() {
         int templateIndex = -1;

         @Override
         public boolean hasNext() {
            return templateIndex + 1 < featureTemplates.length;
         }

         @Override
         public String next() {
            templateIndex++;
            if (templateIndex >= featureTemplates.length) {
               throw new NoSuchElementException();
            }
            int[] template = featureTemplates[templateIndex];
            StringBuilder builder = new StringBuilder();
            for (int element : template) {
               appendTo(builder, "T[" + element + "]=" + iterator.getContextLabel(element).orElse(Sequence.BOS));
            }
            return INTERNER.intern(builder.toString());
         }
      };
   }

   /**
    * Extract iterator.
    *
    * @param result the result
    * @param index  the index
    * @return the iterator
    */
   public Iterator<String> extract(final Labeling result, final int index) {
      return new Iterator<String>() {
         int templateIndex = -1;

         @Override
         public boolean hasNext() {
            return templateIndex + 1 < featureTemplates.length;
         }

         @Override
         public String next() {
            templateIndex++;
            if (templateIndex >= featureTemplates.length) {
               throw new NoSuchElementException();
            }
            int[] template = featureTemplates[templateIndex];
            StringBuilder builder = new StringBuilder();
            for (int element : template) {
               appendTo(builder, "T[" + element + "]=" + result.getLabel(index + element));
            }
            return INTERNER.intern(builder.toString());
         }
      };
   }

   private String label(DecoderState state, int back) {
      back = Math.abs(back) - 1;
      while (back > 0 && state != null) {
         back--;
         state = state.previousState;
      }
      if (state == null || state.tag == null) {
         return Sequence.BOS;
      }
      return state.tag;
   }

   /**
    * Extract iterator.
    *
    * @param prevState the prev state
    * @return the iterator
    */
   public Iterator<String> extract(final DecoderState prevState) {
      return new Iterator<String>() {
         int templateIndex = -1;

         @Override
         public boolean hasNext() {
            return templateIndex + 1 < featureTemplates.length;
         }

         @Override
         public String next() {
            templateIndex++;
            if (templateIndex >= featureTemplates.length) {
               throw new NoSuchElementException();
            }
            int[] template = featureTemplates[templateIndex];
            StringBuilder builder = new StringBuilder();
            for (int element : template) {
               appendTo(builder, "T[" + element + "]=" + label(prevState, element));
            }
            return INTERNER.intern(builder.toString());
         }
      };
   }


   private void appendTo(StringBuilder builder, String feature) {
      if (builder.length() > 0) {
         builder.append("::");
      }
      builder.append(feature);
   }


}// END OF TransitionFeatures
