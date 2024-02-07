package com.example.afinal;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class Menu extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);

        // Obtén una referencia al botón desde tu diseño (layout)
        Button button = findViewById(R.id.button);
        Button frutilla = findViewById(R.id.frutilla);
        Button banana = findViewById(R.id.banana);
        Button uvas = findViewById(R.id.uvas);
        Button mango = findViewById(R.id.mango);

        // Agrega un listener al botón para manejar el evento de clic
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Crea un Intent para iniciar la MainActivity
                Intent intent = new Intent(Menu.this, MainActivity.class);

                // Inicia la MainActivity
                startActivity(intent);
            }
        });

        frutilla.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Crea un Intent para iniciar la MainActivity
                Intent intent = new Intent(Menu.this, Frutilla.class);

                // Inicia la MainActivity
                startActivity(intent);
            }
        });

        banana.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Crea un Intent para iniciar la MainActivity
                Intent intent = new Intent(Menu.this, Banana.class);

                // Inicia la MainActivity
                startActivity(intent);
            }
        });
        uvas.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Crea un Intent para iniciar la MainActivity
                Intent intent = new Intent(Menu.this, Uva.class);

                // Inicia la MainActivity
                startActivity(intent);
            }
        });

        mango.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Crea un Intent para iniciar la MainActivity
                Intent intent = new Intent(Menu.this, Mango.class);

                // Inicia la MainActivity
                startActivity(intent);
            }
        });
    }
}