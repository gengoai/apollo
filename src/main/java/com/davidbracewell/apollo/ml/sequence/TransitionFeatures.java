package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.string.StringUtils;
import com.google.common.base.Preconditions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class TransitionFeatures implements Serializable {
  public static final TransitionFeatures FIRST_ORDER = new TransitionFeatures("-1");
  public static final TransitionFeatures SECOND_ORDER = new TransitionFeatures("-2,-1");
  private final int[][] featureTemplates;
  private int historySize;

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

  public int getHistorySize() {
    return historySize;
  }

  public List<Feature> extract(ContextualIterator<Instance> iterator) {
    List<Feature> features = new ArrayList<>();
    for (int[] template : featureTemplates) {
      StringBuilder builder = new StringBuilder();
      for (int element : template) {
        appendTo(builder, "T[" + element + "]=" + iterator.getContextLabel(element).orElse(Sequence.BOS));
      }
      if (builder.length() > 0) {
        features.add(Feature.TRUE(builder.toString()));
      }
    }
    return features;
  }

  public List<Feature> extract(LabelingResult result, int index) {
    List<Feature> features = new ArrayList<>();
    for (int[] template : featureTemplates) {
      StringBuilder builder = new StringBuilder();
      for (int element : template) {
        appendTo(builder, "T[" + element + "]=" + result.getLabel(index + element));
      }
      if (builder.length() > 0) {
        features.add(Feature.TRUE(builder.toString()));
      }
    }
    return features;
  }

  private String label(DecoderState state, int back) {
    back = Math.abs(back) - 1;
    while (back > 0 && state != null) {
      back--;
      state = state.previousState;
    }
    if (state == null) {
      return Sequence.BOS;
    }
    return state.tag;
  }

  public List<Feature> extract(DecoderState prevState) {
    List<Feature> features = new ArrayList<>();
    for (int[] template : featureTemplates) {
      StringBuilder builder = new StringBuilder();
      for (int element : template) {
        appendTo(builder, "T[" + element + "]=" + label(prevState, element));
      }
      if (builder.length() > 0) {
        features.add(Feature.TRUE(builder.toString()));
      }
    }
    return features;
  }


  private void appendTo(StringBuilder builder, String feature) {
    if (builder.length() > 0) {
      builder.append("::");
    }
    builder.append(feature);
  }


}// END OF TransitionFeatures
