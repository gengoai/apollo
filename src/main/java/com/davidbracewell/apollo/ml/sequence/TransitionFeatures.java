package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.Interner;
import com.davidbracewell.string.StringUtils;
import com.google.common.base.Preconditions;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * @author David B. Bracewell
 */
public class TransitionFeatures implements Serializable {
  public static final TransitionFeatures FIRST_ORDER = new TransitionFeatures("-1");
  public static final TransitionFeatures SECOND_ORDER = new TransitionFeatures("-2,-1");
  private static final Interner<String> INTERNER = new Interner<>();
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


  public Iterator<String> extract(final ContextualIterator<Instance> iterator) {
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

  public Iterator<String> extract(final LabelingResult result, final int index) {
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

  private String label(DecoderRow state, int back) {
    back = state.size() - Math.abs(back) - 1;
    if(back < 0 ){
      return Sequence.BOS;
    }
    return state.getLabel(back);
  }

  public Iterator<String> extract(final DecoderRow row) {
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
          appendTo(builder, "T[" + element + "]=" + label(row, element));
        }
        return INTERNER.intern(builder.toString());
      }
    };
  }

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
